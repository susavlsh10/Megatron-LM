# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parity tests for the cache-aware GDP inference phase boundaries."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.utils import InferenceMode
from megatron.core.ssm import gated_delta_product as gdp_module
from tests.unit_tests.ssm.test_gdp_dynamic_inference import (
    _build_model,
    requires_cuda,
    requires_gdp_model,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = [pytest.mark.internal, requires_cuda, *requires_gdp_model]


class _DecodeContext:
    """The dynamic-context fields read by a one-token GDP decode step."""

    def __init__(self, conv_state, ssm_state, batch_indices, layer_number):
        batch_size = batch_indices.numel()
        self.padded_batch_dimensions = InferenceBatchDimensions(
            token_count=batch_size, decode_req_count=batch_size
        )
        self.num_speculative_tokens = 0
        self.mamba_metadata = SimpleNamespace(batch_indices_decode=batch_indices)
        self.conv_state = conv_state
        self.ssm_state = ssm_state
        self.layer_number = layer_number

    def is_dynamic_batching(self):
        return True

    def is_decode_only(self):
        return self.padded_batch_dimensions.prefill_req_count == 0

    def mamba_states_cache(self, layer_number, intermediate=False):
        assert layer_number == self.layer_number
        assert not intermediate
        return self.conv_state, self.ssm_state


@pytest.fixture
def decode_fixture():
    Utils.initialize_model_parallel(1, 1)
    try:
        model = _build_model(tp=1)
        layer = next(layer for layer in model.decoder.layers if hasattr(layer, "mixer"))
        # The fourth row is a graph-padding request; slot 1 is deliberately unused.
        indices = torch.tensor([2, 0, 3, -1], dtype=torch.int64, device="cuda")
        conv, ssm = layer.mixer.allocate_inference_cache(batch_size=4, max_seqlen=128)
        torch.manual_seed(1234)
        conv.normal_(std=0.05)
        ssm.normal_(std=0.05)
        hidden = torch.randn((4, 1, model.config.hidden_size), device="cuda", dtype=torch.bfloat16)
        layer_number = layer.mixer.layer_number - layer.mixer.pp_layer_offset

        def context():
            return _DecodeContext(conv.clone(), ssm.clone(), indices, layer_number)

        yield layer, hidden, context
    finally:
        Utils.destroy_model_parallel()


def _assert_same_state(actual, expected):
    torch.testing.assert_close(actual.conv_state, expected.conv_state, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(actual.ssm_state, expected.ssm_state, atol=5e-3, rtol=5e-3)


@torch.inference_mode()
def test_project_core_post_matches_serial_decode(decode_fixture):
    layer, hidden, context = decode_fixture
    serial_context, split_context = context(), context()
    assert layer.supports_staged_inference(split_context)

    with InferenceMode.active():
        serial = layer(hidden, inference_context=serial_context)
        projection = layer.inference_project(hidden, split_context)
        with (
            patch.object(layer.mixer, "_decode_conv", wraps=layer.mixer._decode_conv) as conv,
            patch.object(
                gdp_module, "gdp_decode_prepare", wraps=gdp_module.gdp_decode_prepare
            ) as prepare,
            patch.object(
                gdp_module,
                "fused_recurrent_gated_delta_rule_update",
                wraps=gdp_module.fused_recurrent_gated_delta_rule_update,
            ) as recurrent,
            patch.object(
                layer.mixer.out_proj, "forward", wraps=layer.mixer.out_proj.forward
            ) as out_proj,
        ):
            core_output = layer.inference_core(projection, split_context)
            split = layer.inference_post(projection, core_output)
        conv.assert_called_once()
        prepare.assert_called_once()
        recurrent.assert_called_once()
        gemm_input = out_proj.call_args.args[0]
        assert gemm_input.stride() == (gemm_input.shape[-1], gemm_input.shape[-1], 1)

    torch.testing.assert_close(split, serial, atol=5e-3, rtol=5e-3)
    _assert_same_state(split_context, serial_context)


@torch.inference_mode()
def test_non_gdp_mixer_cannot_enter_staged_inference(decode_fixture):
    layer, hidden, context = decode_fixture
    mixer = layer.mixer
    try:
        layer.mixer = torch.nn.Identity()
        assert not layer.supports_staged_inference(context())
        with InferenceMode.active(), pytest.raises(NotImplementedError):
            layer.inference_project(hidden, context())
    finally:
        layer.mixer = mixer


@torch.inference_mode()
def test_staged_core_rejects_prefill_before_cache_update(decode_fixture):
    layer, hidden, context = decode_fixture
    prefill_context = context()
    prefill_context.is_decode_only = lambda: False
    saved_conv = prefill_context.conv_state.clone()
    saved_ssm = prefill_context.ssm_state.clone()
    with InferenceMode.active():
        projection = layer.inference_project(hidden, prefill_context)
        with pytest.raises(NotImplementedError, match="pure decode"):
            layer.inference_core(projection, prefill_context)
    torch.testing.assert_close(prefill_context.conv_state, saved_conv, atol=0, rtol=0)
    torch.testing.assert_close(prefill_context.ssm_state, saved_ssm, atol=0, rtol=0)
