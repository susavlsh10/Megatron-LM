# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Residual operations on token rows, with architecture-owned maps and carried state.

Streamwise inference kernels accept K full-width residual streams per token and
preserve activation-dtype rounding between the read, norm, and write operations.
They own neither model parameters nor process groups.
"""
