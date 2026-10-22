# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reference parity, rounding, strided join views, and graph replay for WR fusion."""

from types import SimpleNamespace

import pytest
import torch
import transformer_engine.pytorch as te

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ops.residual.streamwise import (
    streamwise_read_rmsnorm,
    streamwise_write_read_rmsnorm,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayerSubmodules
from megatron.core.ssm.wide_residual_mamba_layer import WideResidualMambaLayer
from megatron.core.tensor_parallel.inference_layers import InferenceLayerNormColumnParallelLinear
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.residual_connection import ResidualConnection
from megatron.core.transformer.streamwise_residual_ops import (
    streamwise_sigmoid_read,
    streamwise_sigmoid_writeback,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from megatron.core.transformer.wide_residual_layer import (
    StreamwiseSigmoidWideResidualConnection,
    WideResidualTransformerLayer,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("rows", [1, 32, 64])
@pytest.mark.parametrize("retention", [False, True])
def test_write_read_rmsnorm_reference_and_graph_replay(rows, retention):
    """Compare both outputs to the existing write followed by read/norm."""
    seeded()
    width, streams = 3072, 3
    residual = torch.randn(rows, 1, width * streams, device="cuda", dtype=torch.bfloat16)
    update = torch.randn(rows, 1, width, device="cuda", dtype=torch.bfloat16)
    write = torch.randn(streams, device="cuda")
    read = torch.randn(streams, device="cuda")
    retain = torch.randn(streams, device="cuda") if retention else None
    norm = te.RMSNorm(width, eps=1e-5, params_dtype=torch.bfloat16, device="cuda").eval()

    def fused(residual, update, write, read, weight):
        return streamwise_write_read_rmsnorm(
            residual,
            update,
            write,
            read,
            weight,
            streams,
            1e-5,
            retention_logits=retain,
            retention_max_forget=0.1,
        )

    previous = InferenceMode.is_active()
    InferenceMode.set_active()
    try:
        with torch.no_grad():
            norm.weight.normal_()
            assert_replays_bit_exact(
                fused,
                (residual, update, write, read, norm.weight.detach()),
                replays=2,
                contention=True,
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fused(residual, update, write, read, norm.weight)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                carried, normalized = fused(residual, update, write, read, norm.weight)
            for _ in range(2):
                residual.normal_()
                update.normal_()
                graph.replay()
                expected = streamwise_sigmoid_writeback(
                    residual,
                    update,
                    write,
                    streams,
                    retention_logits=retain,
                    retention_max_forget=0.1,
                )
                expected_norm = norm(
                    streamwise_sigmoid_read(expected, read, streams, output_dtype=torch.bfloat16)
                )
                # The original tiled write mixes contracted and uncontracted FP32
                # arithmetic after compilation. The fusion uses explicit FMAs;
                # preserve BF16 boundaries and compare at BF16 precision instead
                # of depending on that compiler/layout-specific contraction.
                torch.testing.assert_close(carried, expected, atol=2e-5, rtol=8e-3)
                torch.testing.assert_close(normalized, expected_norm, atol=2e-3, rtol=8e-3)
                print(
                    "WRITE_READ_PARITY",
                    rows,
                    retention,
                    "carry_mismatches",
                    (carried != expected).sum().item(),
                    "carry_max_abs",
                    (carried.float() - expected.float()).abs().max().item(),
                    "norm_mismatches",
                    (normalized != expected_norm).sum().item(),
                    "norm_max_abs",
                    (normalized.float() - expected_norm.float()).abs().max().item(),
                )
    finally:
        if not previous:
            InferenceMode.unset_active()


def test_write_read_mixer_consumer_adapter():
    class AtomicMixer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, **kwargs):
            return x * 0.25, None

    config = TransformerConfig(
        num_layers=1,
        hidden_size=3072,
        num_attention_heads=2,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        transformer_impl="inference_optimized",
        add_bias_linear=False,
        hidden_dropout=0.0,
        wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
    )
    layer = (
        WideResidualMambaLayer(
            config,
            MambaLayerSubmodules(norm=TENorm, mixer=AtomicMixer, mamba_bda=get_bias_dropout_add),
            pg_collection=ProcessGroupCollection(),
        )
        .cuda()
        .eval()
    )
    consumer = (
        StreamwiseSigmoidWideResidualConnection(config, 2, "test", ProcessGroupCollection())
        .cuda()
        .eval()
    )
    norm = te.RMSNorm(3072, eps=1e-5, params_dtype=torch.bfloat16, device="cuda").eval()
    hidden = torch.randn(32, 1, 9216, device="cuda", dtype=torch.bfloat16)
    context = SimpleNamespace(is_decode_only=lambda: True)
    previous = InferenceMode.is_active()
    InferenceMode.set_active()
    try:
        with torch.no_grad():
            branch, residual, state, _ = layer._prepare_mixer_state(hidden)
            expected = layer._apply_mixer_update(layer.mixer(branch), residual, state)
            read, _ = consumer(expected, operation="read", branch_input_dtype=torch.bfloat16)
            carried, normalized = layer.forward_with_write_read_norm(
                hidden, consumer, norm, context
            )
            torch.testing.assert_close(carried, expected, atol=2e-5, rtol=8e-3)
            torch.testing.assert_close(normalized, norm(read), atol=2e-3, rtol=8e-3)
            assert (
                layer.forward_with_write_read_norm(
                    hidden, consumer, norm, SimpleNamespace(is_decode_only=lambda: False)
                )
                is None
            )
            layer.train()
            assert layer.forward_with_write_read_norm(hidden, consumer, norm, context) is None
    finally:
        if not previous:
            InferenceMode.unset_active()


@pytest.mark.parametrize(
    "rows,width,streams,fp32",
    [
        (1, 128, 2, False),
        (7, 513, 5, True),
        (32, 3072, 3, False),
        (64, 3072, 3, True),
        (32, 512, 3, False),
        (1, 4096, 4, False),
    ],
)
@pytest.mark.parametrize("zero_centered", [False, True])
def test_read_rmsnorm_reference_and_replay(rows, width, streams, fp32, zero_centered):
    seeded()
    x = torch.randn(rows, 1, streams * width, device="cuda").to(
        torch.float32 if fp32 else torch.bfloat16
    )
    logits = torch.randn(streams, device="cuda")
    norm = te.RMSNorm(
        width,
        eps=1e-5,
        zero_centered_gamma=zero_centered,
        params_dtype=torch.bfloat16,
        device="cuda",
    ).eval()

    def fused(x, logits, weight):
        return streamwise_read_rmsnorm(x, logits, weight, streams, 1e-5, zero_centered)

    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))
        assert_replays_bit_exact(
            fused, (x, logits, norm.weight.detach()), replays=3, contention=True
        )
        previous = InferenceMode.is_active()
        InferenceMode.set_active()
        try:
            for _ in range(2):
                x.normal_()
                expected = norm(
                    streamwise_sigmoid_read(x, logits, streams, output_dtype=torch.bfloat16)
                )
                actual = fused(x, logits, norm.weight)
                # Different fixed reduction trees can differ by a BF16 rounding unit.
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=8e-3)
                if width == 3072:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        finally:
            if not previous:
                InferenceMode.unset_active()


def test_prepare_write_state_avoids_read_and_preserves_stateful_fallback(monkeypatch):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        wide_residual=WideResidualConfig(num_streams=3),
    )
    connection = StreamwiseSigmoidWideResidualConnection(
        config, 1, "test", ProcessGroupCollection()
    )

    def fail_read(*args, **kwargs):
        raise AssertionError("Preparing streamwise write state must not read the branch")

    monkeypatch.setattr(connection, "_read_with_output_dtype", fail_read)
    residual = torch.randn(2, 24, dtype=torch.bfloat16)
    assert connection.prepare_write_state(residual)[0] is residual
    state = connection.prepare_write_state(residual, fp32_residual_connection=True)
    torch.testing.assert_close(state[0], residual.float(), atol=0, rtol=0)

    class StatefulConnection(ResidualConnection):
        def _read(self, hidden):
            return hidden[..., :8], (hidden.sum(-1, keepdim=True),)

        def _write(self, output, state, *, dropout_probability, training):
            return state[0]

    stateful = StatefulConnection(24, 8)
    state = stateful.prepare_write_state(residual)
    assert state[0] is residual
    torch.testing.assert_close(state[1], residual.sum(-1, keepdim=True), atol=0, rtol=0)


def test_layer_read_norm_inference_and_training_fallback(monkeypatch):
    class Branch(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden, **kwargs):
            return hidden, None

    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=2,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        transformer_impl="inference_optimized",
        add_bias_linear=False,
        hidden_dropout=0.0,
        wide_residual=WideResidualConfig(num_streams=3),
    )
    layer = (
        WideResidualTransformerLayer(
            config,
            TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=Branch,
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm,
                mlp=Branch,
                mlp_bda=get_bias_dropout_add,
            ),
            add_layer_offset=False,
            pg_collection=ProcessGroupCollection(),
        )
        .cuda()
        .eval()
    )
    x = torch.randn(8, 1, 768, device="cuda", dtype=torch.bfloat16)
    calls = []
    fused = layer.residual_connection_mlp._inference_read_rmsnorm

    def record(*args, **kwargs):
        calls.append(1)
        return fused(*args, **kwargs)

    monkeypatch.setattr(layer.residual_connection_mlp, "_inference_read_rmsnorm", record)
    monkeypatch.setattr(layer.residual_connection_self_attn, "_inference_read_rmsnorm", record)
    previous = InferenceMode.is_active()
    InferenceMode.set_active()
    try:
        with torch.no_grad():
            for hook, connection, norm in [
                (
                    layer._run_input_layernorm,
                    layer.residual_connection_self_attn,
                    layer.input_layernorm,
                ),
                (
                    layer._pre_mlp_layernorm_and_residual,
                    layer.residual_connection_mlp,
                    layer.pre_mlp_layernorm,
                ),
            ]:
                branch, _ = connection(x, operation="read", branch_input_dtype=torch.bfloat16)
                expected = norm(branch)
                actual, residual, state = hook(x)
                assert residual is x and state[0] is x
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=8e-3)
            assert len(calls) == 2
            layer.train()
            layer._pre_mlp_layernorm_and_residual(x)
            assert len(calls) == 2
    finally:
        if not previous:
            InferenceMode.unset_active()


def test_shortcut_router_receives_the_fused_normalized_read():
    seeded()
    h, k = 3072, 3
    logits = torch.randn(k, device="cuda")
    norm = te.RMSNorm(h, eps=1e-5, params_dtype=torch.bfloat16, device="cuda").eval()
    read = SimpleNamespace(read_map=lambda **kwargs: logits, num_streams=k)

    def plain_read(x, **kwargs):
        return streamwise_sigmoid_read(x, logits, k, output_dtype=torch.bfloat16)

    block = SimpleNamespace(
        config=SimpleNamespace(hidden_size=h),
        training=False,
        shortcut_residual_read=read,
        shortcut_pre_mlp_layernorm=norm,
        _inference_shortcut_read_norm=streamwise_read_rmsnorm,
        _moe_route_and_preprocess=lambda x, *args: x,
        _read_shortcut_hidden=plain_read,
        _moe_router_preprocess=lambda x, *args: norm(x),
    )
    previous = InferenceMode.is_active()
    InferenceMode.set_active()
    try:
        with torch.no_grad():
            for rows in (32, 128):
                x = torch.randn(rows, 1, k * h, device="cuda", dtype=torch.bfloat16)
                actual = ShortcutMoEBlock._moe_router_preprocess_inference(block, x)
                torch.testing.assert_close(actual, norm(plain_read(x)), atol=0, rtol=0)
    finally:
        if not previous:
            InferenceMode.unset_active()


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("rows", [1, 64, 128])
def test_projection_owned_norm_graph_replay_and_fallback(rows, sequence_parallel):
    """The Mamba/GDP adapter normalizes once and leaves TP gathering in the projection."""
    Utils.initialize_model_parallel(2, 1)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=3072,
            num_attention_heads=2,
            tensor_model_parallel_size=2,
            sequence_parallel=sequence_parallel,
            normalization="RMSNorm",
            params_dtype=torch.bfloat16,
            bf16=True,
            transformer_impl="inference_optimized",
            add_bias_linear=False,
            hidden_dropout=0.0,
            wide_residual=WideResidualConfig(num_streams=3),
        )

        class ProjectionMixer(torch.nn.Module):
            def __init__(self, config, d_model, pg_collection, **kwargs):
                super().__init__()
                self.in_proj = InferenceLayerNormColumnParallelLinear(
                    d_model,
                    64,
                    config=config,
                    init_method=lambda x: torch.nn.init.normal_(x, std=0.02),
                    gather_output=False,
                    bias=False,
                    skip_bias_add=False,
                    is_expert=False,
                    pg_collection=pg_collection,
                )

        layer = (
            WideResidualMambaLayer(
                config,
                MambaLayerSubmodules(
                    norm=IdentityOp, mixer=ProjectionMixer, mamba_bda=get_bias_dropout_add
                ),
                pg_collection=groups,
            )
            .cuda()
            .eval()
        )
        projection = layer.mixer.in_proj
        projection.set_barrier_before_all_gather()
        assert layer._inference_projection_norm
        keys = set(layer.state_dict())
        assert "mixer.in_proj.layer_norm_weight" in keys
        previous = InferenceMode.is_active()
        InferenceMode.set_active()
        try:
            with torch.no_grad():
                projection.layer_norm_weight.normal_()
                hidden = torch.randn(rows, 1, 9216, device="cuda", dtype=torch.bfloat16)

                def run():
                    normalized, residual, state, context = layer._prepare_mixer_state(hidden)
                    assert residual is hidden and state[0] is hidden and context is None
                    return projection(normalized)[0]

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()
                for _ in range(2):
                    hidden.normal_()
                    graph.replay()
                    branch, _ = layer.residual_connection(
                        hidden, operation="read", branch_input_dtype=torch.bfloat16
                    )
                    # Exercise the original projection path as the reference.
                    projection._input_norm_external = False
                    expected = projection(branch)[0]
                    projection.set_input_norm_external()
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                assert set(layer.state_dict()) == keys
                InferenceMode.unset_active()
                branch, _, _, _ = layer._prepare_mixer_state(hidden)
                expected = projection.normalize_input(branch)
                # Outside inference the projection must still normalize for itself.
                output = projection(branch)[0]
                InferenceMode.set_active()
                reference = projection(expected)[0]
                torch.testing.assert_close(output, reference, atol=0, rtol=0)
        finally:
            if not previous:
                InferenceMode.unset_active()
    finally:
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()
