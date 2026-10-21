# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inference all-reduce using explicit groups and existing symmetric-buffer ownership."""

import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton.utils import are_tensors_nvls_eligible
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager


class InferenceAllReduce:
    """Bind the backend once; return replicated output with independent lifetime."""

    def __init__(self, group, disable_nvls=False):
        self.group = group
        self.kernel = None
        if not disable_nvls and group.size() > 1:
            from megatron.core.ops.communication.nvls_all_reduce import multimem_all_reduce

            self.kernel = multimem_all_reduce

    def __call__(self, x, buffer=None):
        if self.group.size() == 1:
            return x
        if (
            self.kernel is not None
            and x.dtype in (torch.bfloat16, torch.float32)
            and x.is_contiguous()
            and are_tensors_nvls_eligible(x)
        ):
            if buffer is None:
                buffer = SymmetricMemoryManager.get_buffer('tp', self.group).maybe_get_tensor(
                    x.shape, x.dtype
                )
            if buffer['handle'] is not None:
                if x.data_ptr() != buffer['tensor'].data_ptr():
                    buffer['tensor'].copy_(x)
                return self.kernel(torch.empty_like(x), buffer['tensor'], buffer['handle'])
        result = x.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(result, group=self.group)
        return result
