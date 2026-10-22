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

    def __call__(
        self,
        x,
        buffer=None,
        *,
        output_start=0,
        output_tokens=None,
        output_dtype=None,
        output_split=None,
    ):
        """Reduce input, optionally returning only a rank-specific token range.

        The common input controls reduction ownership and the NVLS launch grid;
        each rank can request a different range. Symmetric staging is consumed,
        while the returned tensor has independent lifetime.

        FP32 inputs may emit BF16 output, optionally split along the last axis
        into two contiguous tensors. NVLS fuses these stores into the collective;
        the distributed fallback retains the same reduction and cast semantics.
        """
        if output_tokens is None:
            output_tokens = x.shape[0]
        output_dtype = x.dtype if output_dtype is None else output_dtype
        assert output_dtype == x.dtype or (
            x.dtype == torch.float32 and output_dtype == torch.bfloat16
        )
        if output_split is not None:
            assert x.dtype == torch.float32
            assert 0 < output_split < x.shape[-1]
        assert 0 <= output_start <= x.shape[0] - output_tokens
        assert output_tokens >= 0
        local = (
            x
            if output_start == 0 and output_tokens == x.shape[0]
            else x.narrow(0, output_start, output_tokens)
        )
        if self.group.size() == 1:
            return self._format_output(local, output_dtype, output_split)
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
                if output_split is None:
                    output = torch.empty_like(local, dtype=output_dtype)
                    second_output = None
                else:
                    output = torch.empty(
                        (*local.shape[:-1], output_split), dtype=output_dtype, device=x.device
                    )
                    second_output = torch.empty(
                        (*local.shape[:-1], x.shape[-1] - output_split),
                        dtype=output_dtype,
                        device=x.device,
                    )
                return self.kernel(
                    output,
                    buffer['tensor'],
                    buffer['handle'],
                    input_offset=input_offset,
                    second_output=second_output,
                )
        result = x.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(result, group=self.group)
        return self._format_output(
            result.narrow(0, output_start, output_tokens), output_dtype, output_split
        )

    @staticmethod
    def _format_output(x, dtype, split):
        if split is None:
            return x.to(dtype)
        return tuple(
            part.to(dtype).contiguous() for part in x.split((split, x.shape[-1] - split), -1)
        )
