# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parity of staged and atomic dynamic inference attention with real FlashAttention kernels."""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec,
    wide_residual_gated_delta_product_inference_stack_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import HAVE_FA3, HAVE_FA4
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(params=[2, 3, 4, None], ids=["fa2", "fa3", "fa4", "auto"])
def flash_attention_version(request):
    """Exercise each installed inference backend without requiring every package."""
    version = request.param
    if version == 2 and not is_fa_min_version("2.7.3"):
        pytest.skip("dynamic FA2 batching requires FlashAttention 2.7.3")
    if version == 3 and not HAVE_FA3:
        pytest.skip("FlashAttention 3 is required")
    if version is None and not (HAVE_FA4 or HAVE_FA3 or is_fa_min_version("2.7.3")):
        pytest.skip("a supported dynamic FlashAttention backend is required")
    if version == 4 and not HAVE_FA4:
        pytest.skip("the supported FlashAttention 4 package is required")
    return version


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "attention_output_gate,inference_optimized_spec",
    [(False, False), (True, False), (False, True)],
    ids=["te", "te-output-gate", "nano-inference-wide-residual"],
)
@torch.inference_mode()
def test_dynamic_staged_attention_matches_atomic_prefill_and_decode(
    attention_output_gate, inference_optimized_spec, flash_attention_version
):
    """The staged path commits the same RoPE KV cache and returns the same attention result."""
    InferenceMode.unset_active()
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            flash_attention_version=flash_attention_version,
            attention_output_gate=attention_output_gate,
            transformer_impl=(
                "inference_optimized" if inference_optimized_spec else "transformer_engine"
            ),
            normalization="RMSNorm" if inference_optimized_spec else "LayerNorm",
            add_bias_linear=not inference_optimized_spec,
            wide_residual=(
                WideResidualConfig(num_streams=3, learned_retention=True)
                if inference_optimized_spec
                else None
            ),
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        # Nano uses the inference-optimized GDP stack with a wide residual;
        # both it and the ordinary TE stack expose an attention-only layer.
        attention_spec = (
            wide_residual_gated_delta_product_inference_stack_spec.submodules.attention_layer
            if inference_optimized_spec
            else hybrid_stack_spec.submodules.attention_layer
        )
        layer = (
            build_module(
                attention_spec,
                config=config,
                layer_number=1,
                add_layer_offset=False,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
            .cuda()
            .eval()
        )
        assert layer.supports_two_stage_attention()
        rope = RotaryEmbedding(
            kv_channels=config.kv_channels, rotary_percent=1.0, use_cpu_initialization=True
        )
        rotary_pos_emb = rope(64)

        def make_context():
            context = DynamicInferenceContext(
                model_config=config,
                inference_config=InferenceConfig(
                    max_sequence_length=64,
                    buffer_size_gb=0.0625,
                    block_size_tokens=256,
                    max_requests=4,
                    max_tokens=64,
                    use_flashinfer_fused_rope=False,
                    materialize_only_last_token_logits=False,
                ),
            )
            context.add_request(
                DynamicInferenceRequest(
                    request_id=0,
                    prompt_tokens=torch.arange(8, dtype=torch.long, device="cuda"),
                    sampling_params=SamplingParams(num_tokens_to_generate=2),
                )
            )
            context.initialize_attention_state()
            return context

        atomic_context = make_context()
        staged_context = make_context()
        assert not layer.supports_staged_dynamic_inference(staged_context)

        def committed_cache(context, length):
            # The context allocates one 256-token block for this request. Read
            # only committed tokens; unused and dummy cache rows are undefined.
            block_id = int(context.request_to_kv_block_ids[0, 0])
            assert block_id >= 0
            return context.memory_buffer[:, 0, block_id, :length].clone()

        for cache_length in (8, 9):
            assert atomic_context.is_decode_only() == (cache_length == 9)
            assert staged_context.is_decode_only() == (cache_length == 9)
            assert (
                atomic_context.padded_active_token_count == staged_context.padded_active_token_count
            )
            hidden_states = torch.randn(
                atomic_context.padded_active_token_count,
                1,
                config.hidden_size * (3 if inference_optimized_spec else 1),
                dtype=config.params_dtype,
                device="cuda",
            )

            with InferenceMode.active():
                assert layer.supports_staged_dynamic_inference(staged_context)
                atomic_output, atomic_cross_context = layer(
                    hidden_states.clone(),
                    attention_mask=None,
                    inference_context=atomic_context,
                    rotary_pos_emb=rotary_pos_emb,
                )
                projected = layer.forward_inference_project(
                    hidden_states.clone(),
                    attention_mask=None,
                    inference_context=staged_context,
                    rotary_pos_emb=rotary_pos_emb,
                )
                # The cache write is part of projection. The scheduler may
                # launch independent work before the attention core consumes it.
                torch.testing.assert_close(
                    committed_cache(staged_context, cache_length),
                    committed_cache(atomic_context, cache_length),
                    rtol=0,
                    atol=0,
                )
                core = layer.forward_inference_core(projected)
                staged_output, staged_cross_context = layer.forward_inference_post_core(core)

            assert atomic_cross_context is staged_cross_context is None
            real_tokens = atomic_context.active_token_count
            assert real_tokens == staged_context.active_token_count
            torch.testing.assert_close(
                staged_output[:real_tokens], atomic_output[:real_tokens], rtol=0, atol=0
            )

            if cache_length == 8:
                for context in (atomic_context, staged_context):
                    context.update_requests(
                        active_requests_mask=torch.ones(1, dtype=torch.int32),
                        new_tokens=torch.tensor([11], dtype=torch.long),
                    )
                    context.initialize_attention_state()
    finally:
        InferenceMode.unset_active()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "real_requests,graph_requests,long_kv",
    [(33, 64, True), (65, 128, False), (129, 256, False)],
    ids=["33-in-64-long-kv", "65-in-128", "129-in-256"],
)
@torch.inference_mode()
def test_dynamic_staged_decode_padded_graph_replay(
    real_requests, graph_requests, long_kv, flash_attention_version
):
    """Staged attention handles real requests and padding in one captured graph bucket."""
    InferenceMode.unset_active()
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            flash_attention_version=flash_attention_version,
            attention_output_gate=True,
            transformer_impl="inference_optimized",
            normalization="RMSNorm",
            add_bias_linear=False,
            wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        attention_spec = (
            wide_residual_gated_delta_product_inference_stack_spec.submodules.attention_layer
        )
        layer = (
            build_module(
                attention_spec,
                config=config,
                layer_number=1,
                add_layer_offset=False,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
            .cuda()
            .eval()
        )
        rope = RotaryEmbedding(
            kv_channels=config.kv_channels, rotary_percent=1.0, use_cpu_initialization=True
        )(64)
        context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=8192 if long_kv else 64,
                buffer_size_gb=0.0625,
                block_size_tokens=256,
                max_requests=graph_requests,
                max_tokens=max(512, DynamicInferenceContext.round_up_tokens(real_requests * 8)),
                num_cuda_graphs=1,
                cuda_graph_mixed_prefill_count=0,
                use_flashinfer_fused_rope=False,
                materialize_only_last_token_logits=False,
            ),
        )
        assert len(context.cuda_graph_batch_dimensions_list) == 1
        graph_dimensions = context.cuda_graph_batch_dimensions_list[0]
        assert (
            graph_dimensions.token_count,
            graph_dimensions.prefill_req_count,
            graph_dimensions.decode_req_count,
        ) == (graph_requests, 0, graph_requests)
        for request_id in range(real_requests):
            context.add_request(
                DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=torch.arange(8, dtype=torch.long, device="cuda")
                    + request_id * 16,
                    sampling_params=SamplingParams(num_tokens_to_generate=2),
                )
            )
        context.initialize_attention_state()
        prefill_hidden = torch.randn(
            context.padded_active_token_count,
            1,
            3 * config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        with InferenceMode.active():
            layer(
                prefill_hidden, attention_mask=None, inference_context=context, rotary_pos_emb=rope
            )

        context.update_requests(
            active_requests_mask=torch.ones(real_requests, dtype=torch.int32),
            new_tokens=torch.arange(11, 11 + real_requests, dtype=torch.long),
        )
        context.initialize_attention_state()
        assert context.using_cuda_graph_this_step()
        assert context.active_token_count == real_requests
        assert context.padded_batch_dimensions.decode_req_count == graph_requests
        assert context.padding_slice == slice(real_requests, graph_requests)

        _, kv_lengths, _ = context.cu_kv_lengths()
        torch.testing.assert_close(
            kv_lengths[real_requests:], torch.zeros_like(kv_lengths[real_requests:])
        )
        if long_kv:
            # Populate a valid 8K paged-cache read with a repeated physical page.
            # This tests long-KV decode without a costly 8K prefill.
            key_cache, value_cache, block_table = context.key_value_cache(
                layer.self_attention.layer_number
            )
            assert block_table.shape[1] >= 32
            first_pages = block_table[:real_requests, :1].clone()
            assert bool(torch.all(first_pages >= 0))
            block_table[:real_requests].copy_(first_pages.expand_as(block_table[:real_requests]))
            key_cache.normal_()
            value_cache.normal_()
            kv_lengths[:real_requests].fill_(8192)
            _, active_kv_lengths, _ = context.cu_kv_lengths()
            assert active_kv_lengths.data_ptr() == kv_lengths.data_ptr()
            torch.testing.assert_close(
                active_kv_lengths[:real_requests],
                torch.full_like(active_kv_lengths[:real_requests], 8192),
            )
            assert (
                context.key_value_cache(layer.self_attention.layer_number)[2].data_ptr()
                == block_table.data_ptr()
            )

        decode_hidden = torch.randn(
            graph_requests, 1, 3 * config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        with InferenceMode.active():
            assert layer.supports_staged_dynamic_inference(context)
            if long_kv:
                _, active_kv_lengths, _ = context.cu_kv_lengths()
                assert active_kv_lengths.data_ptr() == kv_lengths.data_ptr()
                torch.testing.assert_close(
                    active_kv_lengths[:real_requests],
                    torch.full_like(active_kv_lengths[:real_requests], 8192),
                )
            atomic_output, _ = layer(
                decode_hidden, attention_mask=None, inference_context=context, rotary_pos_emb=rope
            )

            def run_staged():
                projected = layer.forward_inference_project(
                    decode_hidden,
                    attention_mask=None,
                    inference_context=context,
                    rotary_pos_emb=rope,
                )
                core = layer.forward_inference_core(projected)
                output, _ = layer.forward_inference_post_core(core)
                return output

            staged_output = run_staged()
            torch.testing.assert_close(
                staged_output[:real_requests], atomic_output[:real_requests], rtol=1e-2, atol=1e-2
            )
            assert bool(torch.isfinite(staged_output[:real_requests]).all())

            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    run_staged()
            torch.cuda.current_stream().wait_stream(warmup_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = run_staged()
            graph.replay()
            torch.testing.assert_close(
                graph_output[:real_requests], atomic_output[:real_requests], rtol=1e-2, atol=1e-2
            )

    finally:
        InferenceMode.unset_active()
        Utils.destroy_model_parallel()
