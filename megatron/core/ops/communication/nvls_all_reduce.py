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
def _multimem_all_reduce_kernel(
    output,
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
            source = local_input.to(tl.pointer_type(tl.uint64))
            target = output.to(tl.pointer_type(tl.uint64))
            x, y, z, w = ld_128(source + global_offsets * 2, mask, False, cache_global=True)
            st_128(target + target_offsets * 2, x, y, z, w, mask, False)
            start += tl.num_programs(0) * BLOCK_SIZE


def multimem_all_reduce(output, input_tensor, handle, input_offset=0):
    """Consume symmetric staging and return a contiguous range of its reduction.

    The input begins at the handle's allocation base. ``input_offset`` is in
    elements. All ranks launch the same grid derived from the full input even
    when their output ranges differ, including asymmetric branch TP sizes.
    """
    assert input_tensor.dtype in (torch.bfloat16, torch.float32)
    assert input_tensor.dtype == output.dtype
    assert input_tensor.is_contiguous() and output.is_contiguous()
    assert 0 <= input_offset <= input_tensor.numel() - output.numel()
    pack_width, blocks, config = _kernel_launch_config(
        input_tensor.element_size(), input_tensor.numel(), handle.world_size
    )
    assert (
        input_tensor.numel() % pack_width
        == input_offset % pack_width
        == output.numel() % pack_width
        == 0
    )
    _multimem_all_reduce_kernel[(blocks, 1, 1)](
        output.data_ptr(),
        input_tensor.data_ptr(),
        handle.multicast_ptr,
        handle.signal_pad_ptrs_dev,
        input_tensor.numel() // pack_width,
        output.numel() // pack_width,
        input_offset // pack_width,
        BLOCK_SIZE=config['BLOCK_SIZE'],
        RANK=handle.rank,
        WORLD_SIZE=handle.world_size,
        REDUCE_F32=input_tensor.element_size() == 4,
        num_warps=config['num_warps'],
    )
    return output
