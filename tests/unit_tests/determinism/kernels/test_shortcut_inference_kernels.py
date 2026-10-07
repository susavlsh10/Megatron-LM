# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay the cache-aware GDP, attention and expert adapters used by Shortcut MoE."""

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import (
    wide_residual_gated_delta_product_inference_stack_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.token_dispatcher_inference import (
    InferenceAllGatherDispatcherBase,
    NVLSAllGatherVDispatcher,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact
from tests.unit_tests.ssm.test_gdp_dynamic_inference import requires_gdp_model
from tests.unit_tests.ssm.test_gdp_staged_inference import decode_fixture  # noqa: F401
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    *requires_gdp_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("contention", [False, True])
def test_staged_gdp_output_and_cache_replay(decode_fixture, contention):
    """Fresh cache snapshots must give identical output and state under stream contention."""
    layer, hidden, make_context = decode_fixture

    def run(hidden):
        context = make_context()
        projection = layer.inference_project(hidden, context)
        output = layer.inference_post(projection, layer.inference_core(projection, context))
        return output, context.conv_state, context.ssm_state

    with torch.inference_mode(), InferenceMode.active():
        assert_replays_bit_exact(
            run, (hidden,), backward=False, contention=contention, what="Shortcut GDP decode"
        )


@pytest.fixture
def wide_inference_layers():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_pre_softmax=True,
            moe_router_dtype="fp32",
            moe_shortcut_connection=True,
            expert_tensor_parallel_size=1,
            transformer_impl="inference_optimized",
            inference_shortcut_moe_overlap=True,
            inference_grouped_gemm_backend="vllm",
            inference_moe_token_dispatcher_type="nvls",
            normalization="RMSNorm",
            add_bias_linear=False,
            activation_func=squared_relu,
            params_dtype=torch.bfloat16,
            bf16=True,
            flash_attention_version=2,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
        )
        submodules = wide_residual_gated_delta_product_inference_stack_spec.submodules

        def layer(spec, number):
            return (
                build_module(
                    spec,
                    config=config,
                    layer_number=number,
                    add_layer_offset=False,
                    pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                )
                .cuda()
                .eval()
            )

        yield config, layer(submodules.attention_layer, 1), layer(submodules.moe_layer, 2)
    finally:
        Utils.destroy_model_parallel()


def test_staged_attention_output_and_kv_cache_replay(wide_inference_layers):
    """Replaying projection/core/post overwrites exactly the same committed KV cache."""
    config, layer, _ = wide_inference_layers
    context = DynamicInferenceContext(
        model_config=config,
        inference_config=InferenceConfig(
            max_sequence_length=64,
            buffer_size_gb=0.0625,
            block_size_tokens=256,
            max_requests=4,
            max_tokens=64,
            materialize_only_last_token_logits=False,
            use_flashinfer_fused_rope=False,
        ),
    )
    context.add_request(
        DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=torch.arange(8, device="cuda"),
            sampling_params=SamplingParams(num_tokens_to_generate=2),
        )
    )
    context.initialize_attention_state()
    block = int(context.request_to_kv_block_ids[0, 0])
    hidden = torch.randn(
        context.padded_active_token_count, 1, 384, device="cuda", dtype=torch.bfloat16
    )

    def run(hidden):
        projected = layer.forward_inference_project(hidden, None, inference_context=context)
        output, _ = layer.forward_inference_post_core(layer.forward_inference_core(projected))
        return output[:8], context.memory_buffer[:, 0, block, :8].clone()

    with torch.inference_mode(), InferenceMode.active():
        assert_replays_bit_exact(
            run, (hidden,), backward=False, contention=True, what="Shortcut attention and KV cache"
        )


def test_staged_expert_adapter_replay(wide_inference_layers, monkeypatch):
    """The owning expert module replays its capped BF16 GEMMs and top-k sum bit-exactly."""
    config, _, layer = wide_inference_layers
    experts = layer.mlp.experts
    tokens = 32
    hidden = torch.randn(tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(tokens, 2, device="cuda")
    routes = torch.randint(0, 4, (tokens, 2), device="cuda")
    valid = torch.tensor(tokens, device="cuda", dtype=torch.int32)
    output = torch.empty(tokens, config.hidden_size, device="cuda", dtype=torch.float32)
    # Isolate the expert adapter from multi-rank communication. All compute
    # kernels and real expert weights run; the supplied buffers are the same
    # graph-safe dispatcher contract exercised by the end-to-end EP4 checks.
    monkeypatch.setattr(InferenceAllGatherDispatcherBase, "_valid_tokens", lambda: valid)
    monkeypatch.setattr(
        InferenceAllGatherDispatcherBase, "_get_host_valid_tokens_estimate", lambda: tokens
    )
    monkeypatch.setattr(NVLSAllGatherVDispatcher, "_get_rsv_tensor", lambda: output)

    def run(hidden, probs):
        prepared = experts.prepare_shortcut_overlap(hidden, probs, routes, max_blocks=2)
        result, bias = experts.finish_shortcut_overlap(experts.compute_shortcut_overlap(prepared))
        assert bias is None
        return result

    with torch.inference_mode(), InferenceMode.active():
        assert_replays_bit_exact(
            run, (hidden, probs), backward=False, contention=True, what="Shortcut expert adapter"
        )
