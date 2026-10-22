# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVLS reduction with symmetric staging and independent local output ranges.

Each pack has one reduction owner, which multicasts the result. Ranks then
copy their requested ranges locally, preserving identical replicated values.
The staging input is consumed and may be overwritten by the collective.
"""

import torch
import triton
import triton.language as tl

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.collectives import (
    _kernel_launch_config,
)
from megatron.core.inference.communication.torch_symm_triton.multimem_asm import ld_128, st_128
from megatron.core.inference.communication.torch_symm_triton.utils import get_flat_tid, sync_threads


@triton.jit
def _store_model_value(pointer, value, mask, BF16: tl.constexpr):
    # This collective uses physical thread IDs, not Triton block-vector lanes.
    # Keep stores in per-thread assembly, like ld_128/st_128, to avoid a
    # compiler-inserted leader-thread predicate on an apparently scalar store.
    if BF16:
        tl.inline_asm_elementwise(
            "{ .reg .pred p; .reg .b16 v; setp.ne.u32 p, $3, 0; "
            "cvt.rn.bf16.f32 v, $2; @p st.global.b16 [$1], v; mov.u32 $0, 0; }",
            "=r,l,f,r",
            [pointer, value, mask.to(tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        tl.inline_asm_elementwise(
            "{ .reg .pred p; setp.ne.u32 p, $3, 0; " "@p st.global.f32 [$1], $2; mov.u32 $0, 0; }",
            "=r,l,f,r",
            [pointer, value, mask.to(tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _multimem_all_reduce_kernel(
    output,
    second_output,
    local_input,
    multicast,
    signals,
    input_packs,
    output_packs,
    input_pack_offset,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    REDUCE_F32: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # Pattern 0 publishes writes from preceding GEMM/copy/packing kernels.
    symm_mem_sync(signals, None, RANK, WORLD_SIZE, False, True)
    sync_threads()
    owned_packs = tl.cdiv(input_packs, WORLD_SIZE)
    start = tl.program_id(0) * BLOCK_SIZE
    while start < owned_packs:
        offsets = start + get_flat_tid()
        global_offsets = RANK * owned_packs + offsets
        mask = (offsets < owned_packs) & (global_offsets < input_packs)
        pointers = multicast.to(tl.pointer_type(tl.uint64)) + global_offsets * 2
        x, y, z, w = ld_128(pointers, mask, True, REDUCE_F32)
        # Only this rank reads these packs, so replacing them on every peer is
        # safe. Every replica receives the same single reduction result.
        st_128(pointers, x, y, z, w, mask, True)
        start += tl.num_programs(0) * BLOCK_SIZE
    sync_threads()
    # Pattern 1 publishes each block's multicast stores to its peers.
    symm_mem_sync(signals, None, RANK, WORLD_SIZE, True, True)
    sync_threads()

    # Block b reads only packs broadcast by block b on the other ranks. Local
    # reads after publication need no further cross-rank reuse barrier: later
    # kernels overwrite only their own local staging until the next rendezvous.
    for peer in range(WORLD_SIZE):
        start = tl.program_id(0) * BLOCK_SIZE
        while start < owned_packs:
            offsets = start + get_flat_tid()
            global_offsets = peer * owned_packs + offsets
            target_offsets = global_offsets - input_pack_offset
            mask = (
                (offsets < owned_packs)
                & (global_offsets < input_packs)
                & (target_offsets >= 0)
                & (target_offsets < output_packs)
            )
            if SPLIT > 0 or (REDUCE_F32 and OUTPUT_BF16):
                # Retain FP32 staging and publication barriers. Only the independent
                # output store changes dtype/layout; no residual arithmetic is moved.
                source = local_input.to(tl.pointer_type(tl.uint64))
                values = ld_128(source + global_offsets * 2, mask, False, cache_global=True)
                for component in tl.static_range(4):
                    target_element = target_offsets * 4 + component
                    if SPLIT > 0:
                        row = target_element // ROW_WIDTH
                        column = target_element % ROW_WIDTH
                        pointer = tl.where(
                            column < SPLIT,
                            output + (row * SPLIT + column) * (2 if OUTPUT_BF16 else 4),
                            second_output
                            + (row * (ROW_WIDTH - SPLIT) + column - SPLIT)
                            * (2 if OUTPUT_BF16 else 4),
                        )
                    else:
                        pointer = output + target_element * (2 if OUTPUT_BF16 else 4)
                    _store_model_value(
                        pointer, values[component].to(tl.float32, bitcast=True), mask, OUTPUT_BF16
                    )
            else:
                source = local_input.to(tl.pointer_type(tl.uint64))
                target = output.to(tl.pointer_type(tl.uint64))
                x, y, z, w = ld_128(source + global_offsets * 2, mask, False, cache_global=True)
                st_128(target + target_offsets * 2, x, y, z, w, mask, False)
            start += tl.num_programs(0) * BLOCK_SIZE


def multimem_all_reduce(output, input_tensor, handle, input_offset=0, second_output=None):
    """Consume symmetric staging and return a contiguous range of its reduction.

    The input begins at the handle's allocation base. ``input_offset`` is in
    elements. All ranks launch the same grid derived from the full input even
    when their output ranges differ, including asymmetric branch TP sizes.
    """
    assert input_tensor.dtype in (torch.bfloat16, torch.float32)
    assert input_tensor.dtype == output.dtype or (
        input_tensor.dtype == torch.float32 and output.dtype == torch.bfloat16
    )
    split = output.shape[-1] if second_output is not None else 0
    output_numel = output.numel()
    if second_output is not None:
        assert input_tensor.dtype == torch.float32
        assert second_output.dtype == output.dtype and second_output.is_contiguous()
        assert second_output.shape[:-1] == output.shape[:-1]
        assert split + second_output.shape[-1] == input_tensor.shape[-1]
        output_numel += second_output.numel()
    assert input_tensor.is_contiguous() and output.is_contiguous()
    assert 0 <= input_offset <= input_tensor.numel() - output_numel
    pack_width, blocks, config = _kernel_launch_config(
        input_tensor.element_size(), input_tensor.numel(), handle.world_size
    )
    assert (
        input_tensor.numel() % pack_width
        == input_offset % pack_width
        == output_numel % pack_width
        == 0
    )
    _multimem_all_reduce_kernel[(blocks, 1, 1)](
        output.data_ptr(),
        second_output.data_ptr() if second_output is not None else output.data_ptr(),
        input_tensor.data_ptr(),
        handle.multicast_ptr,
        handle.signal_pad_ptrs_dev,
        input_tensor.numel() // pack_width,
        output_numel // pack_width,
        input_offset // pack_width,
        BLOCK_SIZE=config['BLOCK_SIZE'],
        RANK=handle.rank,
        WORLD_SIZE=handle.world_size,
        REDUCE_F32=input_tensor.element_size() == 4,
        OUTPUT_BF16=output.dtype == torch.bfloat16,
        ROW_WIDTH=input_tensor.shape[-1],
        SPLIT=split,
        num_warps=config['num_warps'],
    )
    return output if second_output is None else (output, second_output)
