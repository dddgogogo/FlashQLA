# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import tilelang


def get_compute_version() -> str:
    return tilelang.contrib.nvcc.get_target_compute_version()


def is_sm90() -> bool:
    return get_compute_version() == "9.0"


def is_sm12x() -> bool:
    return get_compute_version().startswith("12.")
