# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inference-only streamwise residual fusion with explicit BF16 rounding points."""

import torch
import triton
import triton.language as tl


# Native-TE reduction arithmetic adapted from NVIDIA/Megatron-LM PR8351,
# commit bb9567d (BF16 residual-add/RMSNorm). Keep the read-specific producer here.
@triton.jit
def _vector_elements(value, THREADS: tl.constexpr, ELEMENTS: tl.constexpr):
    # Unpack each thread's contiguous BF16 values without a cross-thread
    # gather. Preserve TE's sequential FMA order; a tree over these values is
    # not equivalent. The split operations only rearrange registers.
    if ELEMENTS == 4:
        even, odd = tl.split(tl.reshape(value, (THREADS, 2, 2)))
        v0, v2 = tl.split(even)
        v1, v3 = tl.split(odd)
        return v0, v1, v2, v3
    else:
        even, odd = tl.split(tl.reshape(value, (THREADS, 4, 2)))
        even_low, even_high = tl.split(tl.reshape(even, (THREADS, 2, 2)))
        odd_low, odd_high = tl.split(tl.reshape(odd, (THREADS, 2, 2)))
        v0, v4 = tl.split(even_low)
        v2, v6 = tl.split(even_high)
        v1, v5 = tl.split(odd_low)
        v3, v7 = tl.split(odd_high)
        return v0, v1, v2, v3, v4, v5, v6, v7


@triton.jit
def _warp_allreduce(value, THREADS: tl.constexpr):
    # TE's single-warp allreduce is XOR 1,2,4,8,16, unlike its multi-warp
    # reduction's shuffle-down 16,8,4,2,1. Association is observable in BF16.
    for step in tl.static_range(5):
        value += tl.gather(value, tl.arange(0, THREADS) ^ (1 << step), 0)
    return value


@triton.jit
def _warp_leaders(value, THREADS: tl.constexpr):
    values = tl.reshape(value, (THREADS // 32, 32))
    return tl.sum(tl.where(tl.arange(0, 32)[None, :] == 0, values, 0), 1)


@triton.jit
def _tuned_inverse_rms(chunks, HIDDEN: tl.constexpr, THREADS: tl.constexpr, EPS: tl.constexpr):
    partial = tl.full((THREADS,), 0, tl.float32)
    for chunk in tl.static_range(len(chunks)):
        values = _vector_elements(chunks[chunk], THREADS, 8)
        for element in tl.static_range(8):
            partial += values[element]
    mean = _warp_allreduce(partial, THREADS) * (THREADS / (32.0 * HIDDEN))
    partial = tl.full((THREADS,), 0, tl.float32)
    for chunk in tl.static_range(len(chunks)):
        values = _vector_elements(chunks[chunk], THREADS, 8)
        for element in tl.static_range(8):
            delta = values[element] - mean
            partial = tl.fma(delta, delta, partial)
    m2 = _warp_leaders(_warp_allreduce(partial, THREADS), THREADS)
    means = _warp_leaders(mean, THREADS)
    if THREADS == 128:
        # TE merges four equal-size warp populations using Chan's formulas,
        # shuffle-down 2 then 1. Preserve the CUDA expression's explicit FMAs.
        count = tl.full((4,), HIDDEN / 4.0, tl.float32)
        for step in tl.static_range(2, 0, -1):
            offset = 1 << (step - 1)
            index = tl.minimum(tl.arange(0, 4) + offset, 3)
            other_mean = tl.gather(means, index, 0)
            other_m2 = tl.gather(m2, index, 0)
            other_count = tl.gather(count, index, 0)
            combined_count = count + other_count
            reciprocal = 1.0 / combined_count
            delta = means - other_mean
            correction = ((delta * delta) * count) * other_count
            m2 = tl.fma(correction, reciprocal, m2 + other_m2)
            means = tl.fma(count, means, other_count * other_mean) * reciprocal
            count = combined_count
        mean_scalar = tl.sum(tl.where(tl.arange(0, 4) == 0, means, 0), 0)
        m2_scalar = tl.sum(tl.where(tl.arange(0, 4) == 0, m2, 0), 0)
    else:
        mean_scalar = tl.sum(means, 0)
        m2_scalar = tl.sum(m2, 0)
    return tl.rsqrt(tl.fma(mean_scalar, mean_scalar, (1.0 / HIDDEN) * m2_scalar) + EPS)


@triton.jit
def _normalize_read_chunks(
    chunks,
    partial,
    WEIGHT,
    OUT,
    row,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    THREADS: tl.constexpr,
    ELEMENTS: tl.constexpr,
    TUNED: tl.constexpr,
):
    CHUNKS: tl.constexpr = tl.cdiv(H, THREADS * ELEMENTS)
    columns = tl.arange(0, THREADS)[:, None] * ELEMENTS + tl.arange(0, ELEMENTS)[None, :]
    if TUNED:
        inverse_rms = _tuned_inverse_rms(chunks, H, THREADS, EPS)
    else:
        if THREADS == 128:
            warp_totals = tl.sum(tl.reshape(partial, (4, 32)), 1)
            total = tl.full((), 0, tl.float32)
            for warp in tl.static_range(4):
                total += tl.sum(tl.where(tl.arange(0, 4) == warp, warp_totals, 0), 0)
        else:
            total = tl.sum(_warp_leaders(_warp_allreduce(partial, THREADS), THREADS), 0)
        inverse_rms = tl.rsqrt(tl.fma(total, 1.0 / H, EPS))
    for chunk in tl.static_range(CHUNKS):
        h = columns + chunk * THREADS * ELEMENTS
        weight = tl.load(WEIGHT + h, h < H, 0).to(tl.float32)
        if ZERO_CENTERED:
            weight += 1.0
        tl.store(OUT + row * H + h, (chunks[chunk] * inverse_rms) * weight, h < H)


@triton.jit
def _streamwise_read_rmsnorm_kernel(
    X,
    LOGITS,
    WEIGHT,
    OUT,
    STRIDE: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    THREADS: tl.constexpr,
    ELEMENTS: tl.constexpr,
    TUNED: tl.constexpr,
):
    row = tl.program_id(0)
    CHUNKS: tl.constexpr = tl.cdiv(H, THREADS * ELEMENTS)
    columns = tl.arange(0, THREADS)[:, None] * ELEMENTS + tl.arange(0, ELEMENTS)[None, :]
    partial = tl.full((THREADS,), 0, tl.float32)
    chunks = ()
    for chunk in tl.static_range(CHUNKS):
        h = columns + chunk * THREADS * ELEMENTS
        value = tl.full((THREADS, ELEMENTS), 0, tl.float32)
        for k in tl.static_range(K):
            x = tl.load(X + row * STRIDE + k * H + h, h < H, 0).to(tl.float32)
            factor = tl.sigmoid(tl.load(LOGITS + k).to(tl.float32))
            value = tl.fma(factor, x, value)
        value = value.to(tl.bfloat16).to(tl.float32)
        chunks += (value,)
        if not TUNED:
            values = _vector_elements(value, THREADS, ELEMENTS)
            for element in tl.static_range(ELEMENTS):
                partial = tl.fma(values[element], values[element], partial)
    _normalize_read_chunks(
        chunks, partial, WEIGHT, OUT, row, H, EPS, ZERO_CENTERED, THREADS, ELEMENTS, TUNED
    )


@triton.jit
def _streamwise_write_read_rmsnorm_kernel(
    X,
    UPDATE,
    WRITE_LOGITS,
    RETENTION_LOGITS,
    READ_LOGITS,
    WEIGHT,
    RESIDUAL_OUT,
    OUT,
    H: tl.constexpr,
    K: tl.constexpr,
    HAS_RETENTION: tl.constexpr,
    MAX_FORGET: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    THREADS: tl.constexpr,
    ELEMENTS: tl.constexpr,
    TUNED: tl.constexpr,
):
    row = tl.program_id(0)
    CHUNKS: tl.constexpr = tl.cdiv(H, THREADS * ELEMENTS)
    columns = tl.arange(0, THREADS)[:, None] * ELEMENTS + tl.arange(0, ELEMENTS)[None, :]
    partial = tl.full((THREADS,), 0, tl.float32)
    chunks = ()
    for chunk in tl.static_range(CHUNKS):
        h = columns + chunk * THREADS * ELEMENTS
        update = tl.load(UPDATE + row * H + h, h < H, 0).to(tl.float32)
        value = tl.full((THREADS, ELEMENTS), 0, tl.float32)
        for k in tl.static_range(K):
            offset = row * K * H + k * H + h
            residual = tl.load(X + offset, h < H, 0).to(tl.float32)
            write = 2.0 * tl.sigmoid(tl.load(WRITE_LOGITS + k).to(tl.float32))
            if HAS_RETENTION:
                forget = tl.sigmoid(-tl.load(RETENTION_LOGITS + k).to(tl.float32))
                retention = tl.fma(-MAX_FORGET, forget, 1.0)
                carried = retention * residual
            else:
                carried = residual
            updated = tl.fma(write, update, carried).to(X.dtype.element_ty)
            # Both the carried residual and next read observe the original write's
            # storage rounding. Keep that boundary even though the read uses registers.
            tl.store(RESIDUAL_OUT + offset, updated, h < H)
            read = tl.sigmoid(tl.load(READ_LOGITS + k).to(tl.float32))
            value = tl.fma(read, updated.to(tl.float32), value)
        value = value.to(tl.bfloat16).to(tl.float32)
        chunks += (value,)
        if not TUNED:
            values = _vector_elements(value, THREADS, ELEMENTS)
            for element in tl.static_range(ELEMENTS):
                partial = tl.fma(values[element], values[element], partial)
    _normalize_read_chunks(
        chunks, partial, WEIGHT, OUT, row, H, EPS, ZERO_CENTERED, THREADS, ELEMENTS, TUNED
    )


def streamwise_write_read_rmsnorm(
    residual,
    update,
    write_logits,
    read_logits,
    weight,
    num_streams,
    eps,
    retention_logits=None,
    retention_max_forget=0.0,
    zero_centered=False,
):
    """Write carried streams and normalize their next read in one inference kernel.

    The caller supplies the producer's write parameters and consumer's read/norm
    parameters. Neither input is modified; the updated carry remains materialized.
    """
    tensors = (residual, update, write_logits, read_logits, weight)
    if retention_logits is not None:
        tensors += (retention_logits,)
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError("Streamwise write/read RMSNorm is inference-only.")
    assert residual.is_cuda
    assert all(t.device == residual.device and t.is_contiguous() for t in tensors)
    assert residual.dtype == update.dtype == torch.bfloat16
    assert num_streams > 0
    width = residual.shape[-1] // num_streams
    assert residual.shape[-1] == num_streams * width
    assert update.shape == (*residual.shape[:-1], width)
    assert weight.numel() == width
    assert write_logits.numel() >= num_streams and read_logits.numel() >= num_streams
    assert retention_logits is None or retention_logits.numel() >= num_streams
    carried = torch.empty_like(residual)
    out = torch.empty_like(update)
    rows = residual.numel() // residual.shape[-1]
    if rows:
        _streamwise_write_read_rmsnorm_kernel[(rows,)](
            residual,
            update,
            write_logits,
            retention_logits,
            read_logits,
            weight,
            carried,
            out,
            width,
            num_streams,
            retention_logits is not None,
            retention_max_forget,
            eps,
            zero_centered,
            32 if width <= 2048 else 128,
            4 if width <= 128 else 8,
            width in (512, 768, 1024, 2048, 4096, 8192)
            and rows % 4 == 0
            and weight.data_ptr() % 16 == 0,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return carried, out


def streamwise_read_rmsnorm(x, logits, weight, num_streams, eps, zero_centered=False):
    """Fuse a BF16 streamwise read and RMSNorm on flattenable token rows."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, logits, weight)):
        raise RuntimeError("Streamwise read RMSNorm is inference-only; use torch.no_grad().")
    width = x.shape[-1] // num_streams
    matrix = x.view(-1, x.shape[-1])
    out = torch.empty((*x.shape[:-1], width), dtype=torch.bfloat16, device=x.device)
    if matrix.shape[0]:
        _streamwise_read_rmsnorm_kernel[(matrix.shape[0],)](
            matrix,
            logits,
            weight,
            out,
            matrix.stride(0),
            width,
            num_streams,
            eps,
            zero_centered,
            32 if width <= 2048 else 128,
            4 if width <= 128 else 8,
            width in (512, 768, 1024, 2048, 4096, 8192)
            and matrix.shape[0] % 4 == 0
            and weight.data_ptr() % 16 == 0,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
