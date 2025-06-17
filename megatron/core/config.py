# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

ENABLE_EXPERIMENTAL = False

# `initialize_model_parallel` now supports setting an `attention_data_parallel_size`
# to control subgroups used for attention layers.


def set_experimental_flag(flag: bool):
    """Set the experimental flag to the given value."""
    global ENABLE_EXPERIMENTAL
    ENABLE_EXPERIMENTAL = flag


def is_experimental_enabled():
    """Return the experimental flag."""
    return ENABLE_EXPERIMENTAL
