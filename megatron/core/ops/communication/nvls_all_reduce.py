# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Out-of-place BF16/FP32 all-reduce from a symmetric input to replicated output.

Every rank reads the complete reduced tensor. Matching pre/post signal-pad
barriers protect publication and reuse of the input, including graph replay.
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
    multicast,
    signals,
    packs,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    REDUCE_F32: tl.constexpr,
):
    symm_mem_sync(signals, None, RANK, WORLD_SIZE, True, True)
    sync_threads()
    start = tl.program_id(0) * BLOCK_SIZE
    while start < packs:
        offsets = start + get_flat_tid()
        mask = offsets < packs
        pointers = multicast.to(tl.pointer_type(tl.uint64)) + offsets * 2
        x, y, z, w = ld_128(pointers, mask, True, REDUCE_F32)
        target = output.to(tl.pointer_type(tl.uint64)) + offsets * 2
        st_128(target, x, y, z, w, mask, False)
        start += tl.num_programs(0) * BLOCK_SIZE
    sync_threads()
    symm_mem_sync(signals, None, RANK, WORLD_SIZE, True, True)


def multimem_all_reduce(output, input_tensor, handle):
    """Sum symmetric input across the handle's group into ordinary local output."""
    assert input_tensor.dtype in (torch.bfloat16, torch.float32)
    assert input_tensor.dtype == output.dtype
    assert input_tensor.is_contiguous() and output.is_contiguous()
    assert input_tensor.numel() == output.numel()
    pack_width, blocks, config = _kernel_launch_config(
        input_tensor.element_size(), input_tensor.numel(), 1
    )
    assert input_tensor.numel() % pack_width == 0
    _multimem_all_reduce_kernel[(blocks, 1, 1)](
        output.data_ptr(),
        handle.multicast_ptr,
        handle.signal_pad_ptrs_dev,
        input_tensor.numel() // pack_width,
        BLOCK_SIZE=config['BLOCK_SIZE'],
        RANK=handle.rank,
        WORLD_SIZE=handle.world_size,
        REDUCE_F32=input_tensor.element_size() == 4,
        num_warps=config['num_warps'],
    )
    return output
