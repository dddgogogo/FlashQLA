# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

__version__ = "0.1.0"

__all__ = [
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule",
    "fused_sigmoid_gating_delta_rule_update",
    "flashqla_recurrent_gdn_update",
]

_CHUNK_EXPORTS = {
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule",
}

_RECURRENT_EXPORTS = {
    "fused_sigmoid_gating_delta_rule_update",
    "flashqla_recurrent_gdn_update",
}


def __getattr__(name: str):
    if name in _CHUNK_EXPORTS:
        from flash_qla.ops.gated_delta_rule import chunk as _chunk

        value = getattr(_chunk, name)
    elif name in _RECURRENT_EXPORTS:
        from flash_qla.ops.gated_delta_rule import recurrent_fused as _recurrent

        value = getattr(_recurrent, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
