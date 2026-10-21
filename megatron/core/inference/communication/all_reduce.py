# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inference all-reduce using explicit groups and existing symmetric-buffer ownership."""

import math

import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton.utils import are_tensors_nvls_eligible
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    is_batch_invariant_mode_enabled,
)


class InferenceAllReduce:
    """Bind the backend once; return replicated output with independent lifetime."""

    def __init__(self, group, disable_nvls=False):
        self.group = group
        self.kernel = None
        if not disable_nvls and group.size() > 1:
            from megatron.core.ops.communication.nvls_all_reduce import multimem_all_reduce

            self.kernel = multimem_all_reduce

    def __call__(self, x, buffer=None, *, output_start=0, output_tokens=None):
        """Reduce input, optionally returning only a rank-specific token range.

        The common input controls reduction ownership and the NVLS launch grid;
        each rank can request a different range. Symmetric staging is consumed,
        while the returned tensor has independent lifetime.
        """
        if output_tokens is None:
            output_tokens = x.shape[0]
        assert 0 <= output_start <= x.shape[0] - output_tokens
        assert output_tokens >= 0
        local = (
            x
            if output_start == 0 and output_tokens == x.shape[0]
            else x.narrow(0, output_start, output_tokens)
        )
        if self.group.size() == 1:
            return local
        token_elements = math.prod(x.shape[1:])
        input_offset = output_start * token_elements
        if (
            self.kernel is not None
            and not is_batch_invariant_mode_enabled()
            and x.dtype in (torch.bfloat16, torch.float32)
            and x.is_contiguous()
            and are_tensors_nvls_eligible(x)
            # Range eligibility depends on the common token width, not on a
            # rank-specific range: every rank must select the same backend.
            and (token_elements * x.element_size()) % 16 == 0
        ):
            if buffer is None:
                buffer = SymmetricMemoryManager.get_buffer('tp', self.group).maybe_get_tensor(
                    x.shape, x.dtype
                )
            if buffer['handle'] is not None:
                if x.data_ptr() != buffer['tensor'].data_ptr():
                    buffer['tensor'].copy_(x)
                return self.kernel(
                    torch.empty_like(local),
                    buffer['tensor'],
                    buffer['handle'],
                    input_offset=input_offset,
                )
        result = x.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(result, group=self.group)
        return result.narrow(0, output_start, output_tokens)
