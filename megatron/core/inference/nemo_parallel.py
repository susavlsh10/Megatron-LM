# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Inference placement and activation transitions for independent shortcut branches."""

import math
from copy import copy
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from megatron.core.inference.communication.all_reduce import InferenceAllReduce
from megatron.core.inference.communication.torch_symm_triton.collectives import (
    multimem_all_gather,
    multimem_reduce_scatter,
)
from megatron.core.inference.communication.torch_symm_triton.utils import are_tensors_nvls_eligible
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.parallel_state import create_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)


@dataclass
class NemoParallel:
    """One model replica, with disjoint TP groups for its shortcut branches.

    The outer engine always uses ``full``. Only paired modules use ``branch``;
    no global model-parallel state is changed during construction or execution.
    """

    full: ProcessGroupCollection
    branch: ProcessGroupCollection
    mixer_size: int
    expert_size: int
    is_mixer: bool
    mode: str = "fork"
    join_reduce: object = field(default=None, repr=False)
    state_shapes: dict = field(default_factory=dict)
    attention_heads: dict = field(default_factory=dict)
    model_state_shapes: dict = field(default_factory=dict)
    model_chunking: object = None
    fork_buffer: object = field(default=None, repr=False)
    join_buffer: object = field(default=None, repr=False)

    @property
    def alignment(self):
        return math.lcm(self.full.tp.size(), self.mixer_size, self.expert_size)

    @classmethod
    def create(cls, config, full):
        mixer_size = config.inference_nemo_parallel_mixer_tp_size
        size = full.tp.size()
        expert_size = size - mixer_size
        if not 0 < mixer_size < size:
            raise ValueError("Nemo Parallel requires 0 < mixer TP size < full TP size")
        if full.cp.size() != 1 or full.pp.size() != 1 or full.ep.size() != 1:
            raise ValueError("Nemo Parallel TP currently requires CP=PP=EP=1")
        if full.gtp_remat.size() != 1 or full.expt_gtp_remat.size() != 1:
            raise ValueError("Nemo Parallel currently requires ordinary TP without weight remat")
        mode = getattr(config, "inference_nemo_parallel_join", "fork")
        if mode not in ("fork", "persistent", "replicated"):
            raise ValueError("Unknown Nemo Parallel join layout")
        if config.sequence_parallel != (mode != "replicated"):
            raise ValueError("Only the replicated Nemo layout disables sequence parallelism")
        if config.transformer_impl != "inference_optimized":
            raise ValueError("Nemo Parallel requires optimized inference")
        if not config.moe_shortcut_connection or config.inference_shortcut_moe_overlap:
            raise ValueError("Nemo Parallel requires shortcut pairs without same-GPU overlap")
        if config.fp8 or config.fp4 or not config.bf16:
            raise ValueError("Nemo Parallel currently supports BF16")
        if config.moe_shared_expert_overlap:
            raise ValueError("Nemo Parallel does not support same-GPU shared-expert overlap")
        if getattr(config, "enable_mhc_connections", False):
            raise ValueError("Nemo Parallel does not currently support mHC connections")
        if config.moe_ffn_hidden_size % expert_size:
            raise ValueError("Expert FFN width must be divisible by expert TP size")
        if expert_size > 1 and expert_size & (expert_size - 1):
            raise ValueError("Nemo Parallel expert TP currently requires a power-of-two size")
        ranks = dist.get_process_group_ranks(full.tp)
        # Each replica creates disjoint groups in the same order. Local synchronization
        # avoids involving unrelated DP replicas in group construction.
        mixer = create_group(ranks[:mixer_size], use_local_synchronization=True)
        expert = create_group(ranks[mixer_size:], use_local_synchronization=True)
        is_mixer = full.tp.rank() < mixer_size
        group = mixer if is_mixer else expert
        branch = copy(full)
        for name in ("tp", "tp_cp", "expt_tp", "tp_ep", "tp_ep_pp", "tp_ep_pp_with_egtp_remat"):
            setattr(branch, name, group)
        # Expert replicas follow independent model replicas, not the unused
        # expert-DP coordinates of the outer, uniformly partitioned topology.
        branch.expt_dp = full.dp
        branch.expt_dp_gtp_remat = full.dp_gtp_remat
        return cls(
            full,
            branch,
            mixer_size,
            expert_size,
            is_mixer,
            mode=mode,
            join_reduce=(
                InferenceAllReduce(full.tp, config.inference_disable_triton_nvls_kernels)
                if mode != "fork"
                else None
            ),
        )

    def register_layers(self, layers, layer_types):
        """Record cache requirements from the actual locally owned modules."""
        for index, layer in enumerate(layers):
            if layer is None:
                continue
            attention = getattr(layer, "self_attention", None)
            heads = getattr(attention, "num_query_groups_per_partition", None)
            if heads is not None:
                self.attention_heads[index] = heads
            method = getattr(layer, "mamba_state_shapes_per_request", None)
            if method is None:
                method = getattr(attention, "mamba_state_shapes_per_request", None)
            if method is not None:
                shapes = method()
                if shapes is not None:
                    self.state_shapes[index] = shapes
        from megatron.core.ssm.ssm_inference import ssm_chunking

        # Mixer ranks own every mixer, including the unpaired layers. All ranks
        # need its logical layout even when the expert pool owns no recurrent state.
        layout = [(self.state_shapes, ssm_chunking(layer_types, layers)) if self.is_mixer else None]
        dist.broadcast_object_list(
            layout, src=dist.get_process_group_ranks(self.full.tp)[0], group=self.full.tp
        )
        self.model_state_shapes, self.model_chunking = layout[0]

    def layer_config(self, config):
        result = copy(config)
        result.tensor_model_parallel_size = self.branch.tp.size()
        result.expert_tensor_parallel_size = self.branch.tp.size()
        result.inference_nemo_parallel_role = "mixer" if self.is_mixer else "expert"
        return result

    def initialize_buffers(self, config, max_tokens):
        """Rendezvous the full group before any pool-private symmetric allocations."""
        self.fork_buffer = SymmetricMemoryManager.get_buffer("tp", self.full.tp)
        if not config.inference_disable_triton_nvls_kernels:
            residual_width = config.hidden_size * getattr(config.wide_residual, "num_streams", 1)
            size_bytes = max_tokens * (residual_width + config.hidden_size) * 4
            self.join_buffer = SymmetricMemoryManager.get_buffer(
                "nemo_join", self.full.tp, size_mb=math.ceil(size_bytes / (1024 * 1024))
            )

    def fork(self, hidden, config, already_branch=False):
        """Full-group SP -> branch SP, preserving global token order."""
        if self.mode == "replicated" or already_branch:
            return hidden
        shape = (hidden.shape[0] * self.full.tp.size(), *hidden.shape[1:])
        buffer = (
            self.fork_buffer.maybe_get_tensor(shape, hidden.dtype)
            if self.fork_buffer is not None and not config.inference_disable_triton_nvls_kernels
            else {"handle": None}
        )
        if buffer["handle"] is not None and are_tensors_nvls_eligible(hidden):
            # The full TP buffer may still hold the preceding layer's collective.
            gathered = multimem_all_gather(
                buffer["tensor"], hidden, buffer["handle"], barrier_before=True
            )
        else:
            gathered = gather_from_sequence_parallel_region(hidden, group=self.full.tp)
        tokens = gathered.shape[0]
        if tokens % self.branch.tp.size():
            raise ValueError("Nemo Parallel token bucket must be divisible by both branch sizes")
        count = tokens // self.branch.tp.size()
        start = self.branch.tp.rank() * count
        return gathered.narrow(0, start, count).contiguous()

    def join(self, update, residual, residual_width, config, branch_output=False):
        """Join completed branches and return full-group SP output and mixer residual.

        Every token has one shared contribution, one routed contribution, and
        exactly one residual owner. FP32 transport preserves residual precision;
        the branch sum is rounded back to the model dtype before normalization.
        """
        local_tokens = update.shape[0]
        tokens = local_tokens * (1 if self.mode == "replicated" else self.branch.tp.size())
        shape = (tokens, *update.shape[1:-1], residual_width + update.shape[-1])
        output_shape = (tokens // self.full.tp.size(), *shape[1:])
        buffer = (
            self.join_buffer.maybe_get_tensor(shape, torch.float32)
            if self.join_buffer is not None
            else {"handle": None}
        )
        use_nvls = buffer["handle"] is not None and math.prod(output_shape) % 4 == 0
        if use_nvls:
            # This separate buffer cannot overwrite the fork input while the other
            # pool still reads it. The next fork barriers before the next join reuse.
            payload = buffer["tensor"].zero_()
        else:
            payload = torch.zeros(shape, dtype=torch.float32, device=update.device)
        start = 0 if self.mode == "replicated" else self.branch.tp.rank() * local_tokens
        target = payload.narrow(0, start, local_tokens)
        # Replicated branch results have one owner, independently of pool sizes.
        if self.mode != "replicated" or self.branch.tp.rank() == 0:
            if residual is not None:
                target[..., :residual_width].copy_(residual)
            target[..., residual_width:].copy_(update)
        if self.mode == "replicated" or branch_output:
            joined = self.join_reduce(payload, buffer=buffer)
            if branch_output:
                joined = joined.narrow(0, start, local_tokens)
        elif use_nvls:
            joined = torch.empty(output_shape, dtype=torch.float32, device=update.device)
            multimem_reduce_scatter(joined, payload, buffer["handle"])
        else:
            joined = reduce_scatter_to_sequence_parallel_region(payload, group=self.full.tp)
        return joined[..., residual_width:].to(update.dtype), joined[..., :residual_width]


def forward_shortcut_pair(
    block,
    hidden,
    context,
    attention_mask,
    rotary_pos_emb,
    sequence_len_offset,
    packed_seq_params,
    padding_mask,
):
    """Execute one branch per rank and rejoin before the existing residual write."""
    plan = block.nemo_parallel
    if block.training or context is None or not context.is_dynamic_batching():
        raise RuntimeError("Nemo Parallel is only supported in dynamic inference")
    hidden = plan.fork(hidden, block.config, getattr(block, "nemo_input_branch", False))
    residual_width = hidden.shape[-1]
    if plan.is_mixer:
        computed = block._forward_compute_atomic(
            hidden,
            attention_mask=attention_mask,
            inference_context=context,
            rotary_pos_emb=rotary_pos_emb,
            sequence_len_offset=sequence_len_offset,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        shared, _, residual, _ = block._moe_shared_experts(
            computed, padding_mask=padding_mask, packed_seq_params=packed_seq_params
        )
        if shared is None:
            shared = torch.zeros(
                (*computed.shape[:-1], block.config.hidden_size),
                device=computed.device,
                dtype=computed.dtype,
            )
        update = shared
    else:
        routed = block._read_shortcut_hidden(hidden, recompute_context=None)
        inputs, probs = block._moe_router_preprocess(
            routed, padding_mask=padding_mask, packed_seq_params=packed_seq_params
        )
        inputs, probs = block._launch_dispatch(inputs, probs)
        update, _ = block.moe_layer.mlp.routed_experts_compute(inputs, probs)
        update = block._launch_combine(update)
        update = block.moe_layer.mlp.routed_experts_postprocess(update)
        residual = None
    update, residual = plan.join(
        update,
        residual,
        residual_width,
        block.config,
        branch_output=getattr(block, "nemo_output_branch", False),
    )
    residual = residual.to(torch.float32 if block.config.fp32_residual_connection else hidden.dtype)
    # Residual connection state is architecture-owned. Reuse its read contract rather
    # than assuming a particular wide-residual or ordinary residual representation.
    _, residual, state = block.moe_layer._pre_mlp_layernorm_and_residual(residual)
    return block._postprocess_merged(update, residual, state, packed_seq_params=packed_seq_params)
