# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from .chunk import chunk_gated_delta_rule
from .recurrent_fused import (
    flashqla_recurrent_gdn_update,
    fused_sigmoid_gating_delta_rule_update,
)


__all__ = [
    "chunk_gated_delta_rule",
    "fused_sigmoid_gating_delta_rule_update",
    "flashqla_recurrent_gdn_update",
]
