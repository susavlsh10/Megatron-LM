# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed layout and graph-replay contracts for shortcut branch placement."""

from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.nemo_parallel import NemoParallel
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "mixer_size,disable_nvls,layout",
    [
        (2, True, "fork"),
        (2, False, "fork"),
        (3, False, "fork"),
        (2, False, "persistent"),
        (2, True, "persistent"),
        (3, False, "persistent"),
        (2, False, "replicated"),
    ],
)
def test_branch_layout_and_graph_replay(mixer_size, disable_nvls, layout):
    if Utils.world_size != 4:
        pytest.skip("This topology regression uses four ranks")
    Utils.initialize_model_parallel(
        4, 1, expert_model_parallel_size=1, expert_tensor_parallel_size=4
    )
    try:
        config = SimpleNamespace(
            inference_nemo_parallel_mixer_tp_size=mixer_size,
            sequence_parallel=layout != "replicated",
            inference_nemo_parallel_join=layout,
            transformer_impl="inference_optimized",
            moe_shortcut_connection=True,
            inference_shortcut_moe_overlap=False,
            fp8=None,
            fp4=None,
            bf16=True,
            moe_shared_expert_overlap=False,
            mtp_num_layers=None,
            moe_ffn_hidden_size=96,
            inference_disable_triton_nvls_kernels=disable_nvls,
            hidden_size=8,
            wide_residual=None,
        )
        plan = NemoParallel.create(config, ProcessGroupCollection.use_mpu_process_groups())
        assert plan.mixer_size + plan.expert_size == 4
        if mixer_size == 3 and plan.is_mixer:
            buffer = SymmetricMemoryManager.get_buffer("nemo_test", plan.branch.tp, size_mb=1)
            assert buffer.symm_mem_hdl is None
            assert "power-of-two" in buffer.init_failure_reason
        tokens = 1 if layout == "replicated" else plan.alignment * 2
        plan.initialize_buffers(config, tokens)
        complete = torch.arange(tokens * 8, device="cuda", dtype=torch.float32).view(tokens, 1, 8)
        complete = complete.bfloat16()
        local = (complete if layout == "replicated" else complete.chunk(4)[Utils.rank]).clone()

        def run():
            branch = plan.fork(local, config)
            update = torch.full_like(branch, 2 if plan.is_mixer else 3)
            if layout == "persistent":
                _, residual = plan.join(
                    update, branch if plan.is_mixer else None, 8, config, branch_output=True
                )
                retained = residual.to(branch.dtype)
                retained = plan.fork(retained, config, already_branch=True)
                return branch, plan.join(update, retained if plan.is_mixer else None, 8, config)
            return branch, plan.join(update, branch if plan.is_mixer else None, 8, config)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            branch, (update, residual) = run()
        for delta in (0, 16):
            complete_input = complete + delta
            local.copy_(
                complete_input if layout == "replicated" else complete_input.chunk(4)[Utils.rank]
            )
            graph.replay()
            expected = (
                (complete + delta)
                if layout == "replicated"
                else ((complete + delta).chunk(plan.branch.tp.size())[plan.branch.tp.rank()])
            )
            torch.testing.assert_close(branch, expected, atol=0, rtol=0)
            assert torch.all(update == 5)
            torch.testing.assert_close(residual, local.float(), atol=0, rtol=0)
    finally:
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


def test_symmetric_buffers_are_group_scoped():
    if Utils.world_size != 4:
        pytest.skip("This topology regression uses four ranks")
    Utils.initialize_model_parallel(
        4, 1, expert_model_parallel_size=1, expert_tensor_parallel_size=4
    )
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        full = SymmetricMemoryManager.get_buffer("nemo_test", groups.tp, size_mb=1)
        local = SymmetricMemoryManager.get_buffer("nemo_test", groups.ep, size_mb=1)
        assert full is not local
        assert SymmetricMemoryManager.get_buffer("nemo_test", groups.tp) is full
        with pytest.raises(ValueError, match="ambiguous"):
            SymmetricMemoryManager.get_buffer("nemo_test")
    finally:
        SymmetricMemoryManager.destroy("nemo_test")
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("disable_nvls", [False, True])
def test_replicated_all_reduce_replay(disable_nvls):
    from megatron.core.inference.communication.all_reduce import InferenceAllReduce

    if Utils.world_size != 4:
        pytest.skip("This collective regression uses four ranks")
    Utils.initialize_model_parallel(4, 1)
    try:
        group = ProcessGroupCollection.use_mpu_process_groups().tp
        reduce = InferenceAllReduce(group, disable_nvls)
        for dtype in (torch.bfloat16, torch.float32):
            data = torch.arange(192, device="cuda", dtype=torch.float32).view(12, 16) / 8
            local = (data + Utils.rank).to(dtype)
            start = Utils.rank * 4 if Utils.rank < 3 else 0
            count = 4 if Utils.rank < 3 else 12
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    reduce(local)
                    reduce(local, output_start=start, output_tokens=count)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = reduce(local)
                selected = reduce(local, output_start=start, output_tokens=count)
            for delta in (0, 1):
                local.copy_((data + Utils.rank + delta).to(dtype))
                graph.replay()
                torch.testing.assert_close(
                    result, (data * 4 + 6 + 4 * delta).to(dtype), atol=0, rtol=0
                )
                torch.testing.assert_close(
                    selected,
                    (data * 4 + 6 + 4 * delta).to(dtype).narrow(0, start, count),
                    atol=0,
                    rtol=0,
                )
            # Replicated routing requires identical reduction results on every
            # rank, including non-integer inputs with real rounding decisions.
            generator = torch.Generator(device="cuda").manual_seed(713 + Utils.rank)
            local.copy_(torch.randn(local.shape, device="cuda", generator=generator).to(dtype))
            graph.replay()
            replicas = torch.empty((48, 16), device="cuda", dtype=dtype)
            torch.distributed.all_gather_into_tensor(replicas, result, group=group)
            for replica in replicas.chunk(4):
                torch.testing.assert_close(replica, result, atol=0, rtol=0)
            torch.testing.assert_close(
                selected,
                result.narrow(0, start, count),
                atol=0 if dtype == torch.bfloat16 else 1e-6,
                rtol=0 if dtype == torch.bfloat16 else 1e-6,
            )
    finally:
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("ep_size,expert_tp_size", [(4, 1), (1, 4)])
def test_unsharded_tp1_expert_parallel_dispatch_replay(ep_size, expert_tp_size):
    """TP1 inputs remain distinct across EP or expert TP despite SP being disabled."""
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
    from megatron.core.transformer.transformer_config import TransformerConfig

    if Utils.world_size != 4:
        pytest.skip("This EP compatibility regression uses four ranks")
    Utils.initialize_model_parallel(
        1, 1, expert_model_parallel_size=ep_size, expert_tensor_parallel_size=expert_tp_size
    )
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            num_moe_experts=8,
            moe_router_topk=2,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=expert_tp_size,
            sequence_parallel=False,
            add_bias_linear=False,
        )
        rank = groups.tp_ep.rank()
        local_experts = 8 // ep_size
        first_expert = groups.ep.rank() * local_experts
        dispatcher = NVLSAllGatherVDispatcher(
            local_experts, list(range(first_expert, first_expert + local_experts)), config, groups
        )
        assert not dispatcher.replicated_input
        NVLSAllGatherVDispatcher._delete_buffers()
        NVLSAllGatherVDispatcher.allocate_buffers(8, 2, 32, groups.tp_ep)
        local = torch.full((8, 1, 32), rank + 1, dtype=torch.bfloat16, device="cuda")
        routing = torch.tensor([0, 1], device="cuda").expand(8, 2).clone()
        probs = torch.full((8, 2), 0.5, dtype=torch.float32, device="cuda")

        def run():
            hidden, p = dispatcher.dispatch_preprocess(local, routing, probs)
            gathered, p = dispatcher.token_dispatch(hidden, p)
            combined = dispatcher.token_combine(gathered.float())
            return gathered, dispatcher.combine_postprocess(combined)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gathered, result = run()
        for delta in (0, 1):
            local.fill_(rank + 1 + delta)
            graph.replay()
            expected = (
                torch.arange(1, 5, device="cuda", dtype=torch.bfloat16)
                .add(delta)
                .repeat_interleave(8)
            )
            torch.testing.assert_close(gathered[:, 0], expected, atol=0, rtol=0)
            torch.testing.assert_close(result, local * 4, atol=0, rtol=0)
    finally:
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()


def test_hybrid_checkpoint_resharding_and_caches(tmp_path):
    """Reshard a TP4 checkpoint, check ownership/caches, and compare real logits."""
    if Utils.world_size != 4:
        pytest.skip("This topology regression uses four ranks")
    from megatron.core import dist_checkpointing
    from megatron.core.activations import squared_relu
    from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams
    from megatron.core.inference.utils import InferenceMode
    from megatron.core.models.hybrid.hybrid_layer_specs import (
        gated_delta_product_inference_stack_spec,
        wide_residual_gated_delta_product_inference_stack_spec,
    )
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.wide_residual_config import WideResidualConfig

    Utils.initialize_model_parallel(
        4, 1, expert_model_parallel_size=1, expert_tensor_parallel_size=2
    )
    try:
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=7,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=4,
            ffn_hidden_size=128,
            normalization="RMSNorm",
            bf16=True,
            params_dtype=torch.bfloat16,
            mamba_num_heads=16,
            mamba_head_dim=16,
            mamba_num_groups=4,
            mamba_state_dim=16,
            gdp_num_householder=2,
            is_hybrid_model=True,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            expert_tensor_parallel_size=2,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_pre_softmax=True,
            moe_router_dtype="fp32",
            moe_ffn_hidden_size=128,
            moe_shared_expert_intermediate_size=128,
            moe_latent_size=32,
            moe_use_norm_before_up_proj=True,
            moe_shortcut_connection=True,
            transformer_impl="inference_optimized",
            inference_grouped_gemm_backend="vllm",
            inference_moe_token_dispatcher_type="nvls",
            add_bias_linear=False,
            activation_func=squared_relu,
            wide_residual=WideResidualConfig(num_streams=3),
            inference_nemo_parallel_mixer_tp_size=2,
        )
        groups = ProcessGroupCollection.use_mpu_process_groups()

        def build(model_config, pattern):
            return (
                HybridModel(
                    config=model_config,
                    hybrid_stack_spec=(
                        wide_residual_gated_delta_product_inference_stack_spec
                        if model_config.wide_residual is not None
                        else gated_delta_product_inference_stack_spec
                    ),
                    vocab_size=128,
                    max_sequence_length=64,
                    hybrid_layer_pattern=pattern,
                    pg_collection=groups,
                )
                .cuda()
                .eval()
            )

        def make_context(model, alignment):
            return DynamicInferenceContext(
                model.config,
                InferenceConfig(
                    pg_collection=groups,
                    nemo_parallel=model.decoder.nemo_parallel,
                    mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
                    max_sequence_length=64,
                    max_requests=alignment,
                    max_tokens=alignment * 16,
                    buffer_size_gb=0.0625,
                    block_size_tokens=256,
                    num_cuda_graphs=None,
                    use_flashinfer_fused_rope=False,
                    materialize_only_last_token_logits=False,
                ),
            )

        def forward(model, context):
            context.add_request(
                DynamicInferenceRequest(
                    request_id=0,
                    prompt_tokens=torch.arange(8, device="cuda", dtype=torch.long),
                    sampling_params=SamplingParams(num_tokens_to_generate=1),
                )
            )
            context.initialize_attention_state()
            input_ids, position_ids = context.current_input_and_position_ids()
            with torch.inference_mode(), InferenceMode.active():
                return model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=None,
                    inference_context=context,
                    runtime_gather_output=True,
                ).clone()

        paths = [str(tmp_path)]
        torch.distributed.broadcast_object_list(paths, src=0)
        template = config
        for case, pattern in enumerate(("MEM*E*-", "ME*E-", "ME*E-")):
            model_parallel_cuda_manual_seed(1234 + case)
            config = copy(template)
            config.num_layers = len(pattern)
            if case == 2:
                # A dimension-compatible 3+1 layout exercises real NCCL mixer TP
                # and a single-GPU expert path, not just activation redistribution.
                config.inference_nemo_parallel_mixer_tp_size = 3
                config.wide_residual = None
                config.hidden_size = 192
                config.num_attention_heads = config.num_query_groups = 12
                config.kv_channels = 16
                config.mamba_num_heads = 24
                config.mamba_num_groups = 12
                config.ffn_hidden_size = config.moe_ffn_hidden_size = 192
                config.moe_shared_expert_intermediate_size = 192
            baseline_config = copy(config)
            baseline_config.inference_nemo_parallel_mixer_tp_size = 0
            baseline = build(baseline_config, pattern)
            alignment = 12 if case == 2 else 4
            reference = forward(baseline, make_context(baseline, alignment))
            checkpoint_dir = Path(paths[0]) / str(case)
            checkpoint_dir.mkdir(exist_ok=True)
            dist_checkpointing.save(baseline.sharded_state_dict(), checkpoint_dir)
            for layout in ("fork", "persistent", "replicated"):
                candidate_config = copy(config)
                candidate_config.inference_nemo_parallel_join = layout
                candidate_config.sequence_parallel = layout != "replicated"
                model = build(candidate_config, pattern)
                model.load_state_dict(
                    dist_checkpointing.load(model.sharded_state_dict(), checkpoint_dir)
                )
                from megatron.core.dist_checkpointing.mapping import apply_factories
                from megatron.core.dist_checkpointing.validation import (
                    determine_global_metadata,
                    validate_sharding_integrity,
                )

                checkpoint_state = model.sharded_state_dict()
                apply_factories(checkpoint_state)
                _, global_metadata = determine_global_metadata(checkpoint_state)
                validate_sharding_integrity(global_metadata)
                plan = model.decoder.nemo_parallel
                pair = model.decoder.layers[0]
                assert (pair.compute_layer is not None) == plan.is_mixer
                assert (pair.moe_layer.mlp.experts is None) == plan.is_mixer
                assert (pair.moe_layer.mlp.shared_experts is not None) == plan.is_mixer
                context = make_context(model, alignment)
                if plan.expert_size == 1 and not plan.is_mixer:
                    assert NVLSAllGatherVDispatcher._get_rsv_tensor() is None
                mixer_states = {0, 2} if case == 0 else {0}
                expert_states = {2} if case == 0 else set()
                assert set(plan.state_shapes) == (mixer_states if plan.is_mixer else expert_states)
                assert context.is_hybrid_model and context.gdp_num_householder == 2
                limits = [None] * 4
                torch.distributed.all_gather_object(
                    limits,
                    (
                        context.kv_block_allocator.pool_size,
                        context.max_requests,
                        context.mamba_chunk_size,
                    ),
                )
                assert all(limit == limits[0] for limit in limits)
                assert set(context._nemo_states) == set(plan.state_shapes)
                for layer, (conv_shape, ssm_shape) in plan.state_shapes.items():
                    conv, ssm = context.mamba_states_cache(layer + 1)
                    assert conv.shape == (context.max_requests, *conv_shape)
                    assert ssm.shape == (context.max_requests, *ssm_shape)
                    assert conv.is_contiguous() and ssm.is_contiguous()
                    conv.fill_(1)
                    ssm.fill_(1)
                context._pending_mamba_zeros.append(0)
                context._execute_pending_mamba_ops()
                for layer in plan.state_shapes:
                    conv, ssm = context.mamba_states_cache(layer + 1)
                    assert torch.all(conv[0] == 0) and torch.all(ssm[0] == 0)
                    assert torch.all(conv[1] == 1) and torch.all(ssm[1] == 1)
                logits = forward(model, context)
                # TP layouts can pad to different token counts; compare the real prompt.
                torch.testing.assert_close(logits[:, :8], reference[:, :8], atol=0.01, rtol=0.05)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()
