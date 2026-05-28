import os

import torch
import tilelang
import tilelang.language as T


_RECURRENT_AUTOTUNE_CACHE: dict[tuple, int] = {}
_RECURRENT_KERNEL_CACHE: dict[tuple, object] = {}
_IDENTITY_I32_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def clear_recurrent_autotune_cache() -> None:
    _RECURRENT_AUTOTUNE_CACHE.clear()
    _RECURRENT_KERNEL_CACHE.clear()
    _IDENTITY_I32_CACHE.clear()


def _get_recurrent_kernel(factory, **kwargs):
    key = (
        torch.cuda.current_device(),
        id(factory),
        tuple(sorted(kwargs.items(), key=lambda item: item[0])),
    )
    kernel = _RECURRENT_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = factory(**kwargs)
        _RECURRENT_KERNEL_CACHE[key] = kernel
    return kernel


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in ("0", "false", "off", "no")


# Opt-in: the cached Python dispatch still costs enough to regress tiny kernels.
_RECURRENT_AUTOTUNE_ENABLED = _env_flag("FLASHQLA_RECURRENT_AUTOTUNE", False)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, 0)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(value, 0.0)


def _autotune_enabled() -> bool:
    return _RECURRENT_AUTOTUNE_ENABLED


def _valid_block_dv_candidates(candidates: list[int], head_v_dim: int) -> list[int]:
    valid: list[int] = []
    for block_dv in candidates:
        if block_dv in valid:
            continue
        if block_dv > head_v_dim:
            continue
        if block_dv in (4, 8, 16, 32, 64) or block_dv == head_v_dim:
            valid.append(block_dv)
    return valid


def _autotune_block_dv_candidates(
    num_sequences: int,
    regular_tokens_per_seq: int | None,
    head_v_dim: int,
    cache_intermediate: bool,
    has_tree: bool,
    disable_state_update: bool,
) -> list[int]:
    if regular_tokens_per_seq is None:
        return []
    if cache_intermediate:
        return _valid_block_dv_candidates([4, 8], head_v_dim)
    if (
        regular_tokens_per_seq == 1
        and not has_tree
        and not disable_state_update
    ):
        if num_sequences == 1:
            return _valid_block_dv_candidates([32, 16, head_v_dim], head_v_dim)
        if num_sequences <= 4:
            return _valid_block_dv_candidates([16, 4, 8, 32], head_v_dim)
        return _valid_block_dv_candidates([4, 8, 16, 32], head_v_dim)
    return []


def _autotune_cache_key(
    q: torch.Tensor,
    state: torch.Tensor,
    num_key_heads: int,
    num_value_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    num_sequences: int,
    regular_tokens_per_seq: int | None,
    cache_steps: int,
    cache_intermediate: bool,
    has_tree: bool,
    disable_state_update: bool,
    use_qk_l2norm_in_kernel: bool,
) -> tuple:
    mode = 2 if has_tree else 1 if cache_intermediate else 0
    return (
        q.get_device(),
        mode,
        num_key_heads,
        num_value_heads,
        head_k_dim,
        head_v_dim,
        num_sequences,
        regular_tokens_per_seq,
        cache_steps,
        cache_intermediate,
        has_tree,
        disable_state_update,
        use_qk_l2norm_in_kernel,
        q.dtype,
        state.dtype,
    )


def _benchmark_recurrent_block_dv(
    block_dv: int,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float,
    softplus_beta: float,
    softplus_threshold: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor,
    disable_state_update: bool,
    intermediate: torch.Tensor | None,
    intermediate_state_indices: torch.Tensor,
    cache_steps: int,
    retrieve_parent_token: torch.Tensor | None,
    warmup: int,
    iters: int,
) -> float:
    trial_state = state.detach().clone()
    trial_intermediate = torch.empty_like(intermediate) if intermediate is not None else None

    def call() -> None:
        fused_sigmoid_gating_delta_rule_update(
            A_log,
            a,
            dt_bias,
            q,
            k,
            v,
            b,
            trial_state,
            initial_state_indices,
            scale=scale,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=trial_intermediate,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            block_dv=block_dv,
            assume_regular=True,
        )

    for _ in range(warmup):
        call()
    torch.cuda.synchronize(q.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(q.device)
    start.record(stream)
    for _ in range(iters):
        call()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / max(iters, 1)


def _select_autotuned_block_dv(
    fallback_block_dv: int,
    candidates: list[int],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float,
    softplus_beta: float,
    softplus_threshold: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor,
    disable_state_update: bool,
    intermediate: torch.Tensor | None,
    intermediate_state_indices: torch.Tensor,
    cache_steps: int,
    retrieve_parent_token: torch.Tensor | None,
    cache_key: tuple,
) -> int:
    cached = _RECURRENT_AUTOTUNE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if getattr(torch.cuda, "is_current_stream_capturing", lambda: False)():
        return fallback_block_dv
    if len(candidates) <= 1:
        selected = candidates[0] if candidates else fallback_block_dv
        _RECURRENT_AUTOTUNE_CACHE[cache_key] = selected
        return selected

    warmup = _env_int("FLASHQLA_RECURRENT_AUTOTUNE_WARMUP", 5)
    iters = _env_int("FLASHQLA_RECURRENT_AUTOTUNE_ITERS", 30)
    if iters == 0:
        _RECURRENT_AUTOTUNE_CACHE[cache_key] = fallback_block_dv
        return fallback_block_dv

    candidates = _valid_block_dv_candidates([fallback_block_dv, *candidates], v.shape[-1])
    timings: dict[int, float] = {}
    for candidate in candidates:
        timings[candidate] = _benchmark_recurrent_block_dv(
            candidate,
            A_log,
            a,
            dt_bias,
            q,
            k,
            v,
            b,
            state,
            initial_state_indices,
            scale,
            softplus_beta,
            softplus_threshold,
            use_qk_l2norm_in_kernel,
            cu_seqlens,
            disable_state_update,
            intermediate,
            intermediate_state_indices,
            cache_steps,
            retrieve_parent_token,
            warmup,
            iters,
        )
    measured_best = min(timings, key=timings.get)
    fallback_ms = timings.get(fallback_block_dv)
    margin = _env_float("FLASHQLA_RECURRENT_AUTOTUNE_MARGIN", 0.03)
    if fallback_ms is None or timings[measured_best] < fallback_ms * (1.0 - margin):
        selected = measured_best
    else:
        selected = fallback_block_dv
    _RECURRENT_AUTOTUNE_CACHE[cache_key] = selected
    return selected


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_flashqla_gdn_update(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    disable_state_update,
    cache_intermediate_states,
    has_tree_attention,
    block_DV: int = 128,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    cache_steps = T.dynamic("cache_steps")
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)
    intermediate_shape = (num_sequences, cache_steps, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_update_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        cu_seqlens: T.Tensor((num_sequences + 1,), dtype=indices_dtype),
        intermediate_states_buffer: T.Tensor(intermediate_shape, dtype=state_dtype),
        intermediate_state_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        retrieve_parent_token: T.Tensor((num_sequences, cache_steps), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=128) as (bid,):
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            seq_len = T.alloc_var("int32")
            state_idx = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")

            seq_start = cu_seqlens[bn]
            seq_end = cu_seqlens[bn + 1]
            seq_len = seq_end - seq_start
            state_idx = h0_indices[bn]
            cache_idx = intermediate_state_indices[bn]

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            q_fragment = T.alloc_fragment((DK,), dtype=accum_dtype)
            k_fragment = T.alloc_fragment((DK,), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_DV,), dtype=accum_dtype)
            kv_fragment = T.alloc_fragment((block_DV,), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_DV,), dtype=accum_dtype)
            reduce_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            q_norm = T.alloc_fragment((1,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((1,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            parent_step = T.alloc_fragment((1,), dtype=indices_dtype)
            cache_barrier = T.alloc_barrier(arrive_count=128)

            T.clear(h_fragment)
            if state_idx >= 0:
                for jk, jv in T.Parallel(DK, block_DV):
                    if bv * block_DV + jv < DV:
                        h_fragment[jk, jv] = h0_source[
                            state_idx, bh, bv * block_DV + jv, jk
                        ]

            for step in T.serial(seq_len):
                if has_tree_attention:
                    if step != 0 and cache_idx >= 0:
                        parent_step[0] = retrieve_parent_token[bn, step]
                        for jk, jv in T.Parallel(DK, block_DV):
                            if bv * block_DV + jv < DV:
                                h_fragment[jk, jv] = intermediate_states_buffer[
                                    cache_idx,
                                    parent_step[0],
                                    bh,
                                    bv * block_DV + jv,
                                    jk,
                                ]

                token_idx = seq_start + step

                for jk in T.Parallel(DK):
                    q_fragment[jk] = q[0, token_idx, bhq, jk]
                    k_fragment[jk] = k[0, token_idx, bhq, jk]

                if use_qk_l2norm_in_kernel:
                    q_norm[0] = 0.0
                    k_norm[0] = 0.0
                    for jk in T.serial(DK):
                        q_norm[0] += q_fragment[jk] * q_fragment[jk]
                        k_norm[0] += k_fragment[jk] * k_fragment[jk]
                    q_norm[0] = T.rsqrt(q_norm[0] + 1e-6)
                    k_norm[0] = T.rsqrt(k_norm[0] + 1e-6)
                    for jk in T.Parallel(DK):
                        q_fragment[jk] *= q_norm[0]
                        k_fragment[jk] *= k_norm[0]

                for jk in T.Parallel(DK):
                    q_fragment[jk] *= scale

                a_raw[0] = a[0, token_idx, bh]
                b_raw[0] = b[0, token_idx, bh]
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = T.log(1.0 + T.exp(beta_x[0])) / softplus_beta
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp(-T.exp(A_log[bh]) * softplus_x[0])
                beta[0] = 1.0 / (1.0 + T.exp(-b_raw[0]))

                for jk, jv in T.Parallel(DK, block_DV):
                    h_fragment[jk, jv] *= exp_g[0]

                for jk, jv in T.Parallel(DK, block_DV):
                    if bv * block_DV + jv < DV:
                        reduce_fragment[jk, jv] = h_fragment[jk, jv] * k_fragment[jk]
                    else:
                        reduce_fragment[jk, jv] = 0.0
                T.reduce_sum(reduce_fragment, kv_fragment, dim=0, clear=True)

                for jv in T.Parallel(block_DV):
                    if bv * block_DV + jv < DV:
                        v_fragment[jv] = v[0, token_idx, bh, bv * block_DV + jv]
                        v_fragment[jv] = (v_fragment[jv] - kv_fragment[jv]) * beta[0]

                for jk, jv in T.Parallel(DK, block_DV):
                    if bv * block_DV + jv < DV:
                        h_fragment[jk, jv] += k_fragment[jk] * v_fragment[jv]

                for jk, jv in T.Parallel(DK, block_DV):
                    if bv * block_DV + jv < DV:
                        reduce_fragment[jk, jv] = h_fragment[jk, jv] * q_fragment[jk]
                    else:
                        reduce_fragment[jk, jv] = 0.0
                T.reduce_sum(reduce_fragment, o_fragment, dim=0, clear=True)

                for jv in T.Parallel(block_DV):
                    if bv * block_DV + jv < DV:
                        o[0, token_idx, bh, bv * block_DV + jv] = o_fragment[jv]

                if cache_intermediate_states:
                    if cache_idx >= 0:
                        for jk, jv in T.Parallel(DK, block_DV):
                            if bv * block_DV + jv < DV:
                                intermediate_states_buffer[
                                    cache_idx, step, bh, bv * block_DV + jv, jk
                                ] = h_fragment[jk, jv]
                    if has_tree_attention:
                        T.barrier_arrive(cache_barrier)
                        T.barrier_wait(cache_barrier, step % 2)

            if not disable_state_update:
                if state_idx >= 0:
                    for jk, jv in T.Parallel(DK, block_DV):
                        if bv * block_DV + jv < DV:
                            h0_source[state_idx, bh, bv * block_DV + jv, jk] = (
                                h_fragment[jk, jv]
                            )

    return tilelang_flashqla_gdn_update_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_flashqla_gdn_regular(
    H,
    HV,
    DK,
    DV,
    tokens_per_seq,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    disable_state_update,
    cache_intermediate_states,
    has_tree_attention,
    block_DV: int = 128,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    cache_steps = T.dynamic("cache_steps")
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)
    intermediate_shape = (num_sequences, cache_steps, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_regular_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        intermediate_states_buffer: T.Tensor(intermediate_shape, dtype=state_dtype),
        intermediate_state_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        retrieve_parent_token: T.Tensor((num_sequences, cache_steps), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=128) as (bid,):
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            seq_start = T.alloc_var("int32")
            state_idx = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")

            seq_start = bn * tokens_per_seq
            state_idx = h0_indices[bn]
            cache_idx = -1
            if cache_intermediate_states:
                cache_idx = intermediate_state_indices[bn]

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            q_fragment = T.alloc_fragment((DK,), dtype=accum_dtype)
            k_fragment = T.alloc_fragment((DK,), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_DV,), dtype=accum_dtype)
            kv_fragment = T.alloc_fragment((block_DV,), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_DV,), dtype=accum_dtype)
            reduce_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            q_norm = T.alloc_fragment((1,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((1,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            parent_step = T.alloc_fragment((1,), dtype=indices_dtype)
            cache_barrier = T.alloc_barrier(arrive_count=128)

            T.clear(h_fragment)
            if state_idx >= 0:
                for jk, jv in T.Parallel(DK, block_DV):
                    if bv * block_DV + jv < DV:
                        h_fragment[jk, jv] = h0_source[
                            state_idx, bh, bv * block_DV + jv, jk
                        ]

            for step in T.serial(tokens_per_seq):
                if has_tree_attention:
                    if step != 0 and cache_idx >= 0:
                        parent_step[0] = retrieve_parent_token[bn, step]
                        for jk, jv in T.Parallel(DK, block_DV):
                            if bv * block_DV + jv < DV:
                                h_fragment[jk, jv] = intermediate_states_buffer[
                                    cache_idx,
                                    parent_step[0],
                                    bh,
                                    bv * block_DV + jv,
                                    jk,
                                ]

                token_idx = seq_start + step

                for jk in T.Parallel(DK):
                    q_fragment[jk] = q[0, token_idx, bhq, jk]
                    k_fragment[jk] = k[0, token_idx, bhq, jk]

                if use_qk_l2norm_in_kernel:
                    q_norm[0] = 0.0
                    k_norm[0] = 0.0
                    for jk in T.serial(DK):
                        q_norm[0] += q_fragment[jk] * q_fragment[jk]
                        k_norm[0] += k_fragment[jk] * k_fragment[jk]
                    q_norm[0] = T.rsqrt(q_norm[0] + 1e-6)
                    k_norm[0] = T.rsqrt(k_norm[0] + 1e-6)
                    for jk in T.Parallel(DK):
                        q_fragment[jk] *= q_norm[0]
                        k_fragment[jk] *= k_norm[0]

                for jk in T.Parallel(DK):
                    q_fragment[jk] *= scale

                a_raw[0] = a[0, token_idx, bh]
                b_raw[0] = b[0, token_idx, bh]
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = T.log(1.0 + T.exp(beta_x[0])) / softplus_beta
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp(-T.exp(A_log[bh]) * softplus_x[0])
                beta[0] = 1.0 / (1.0 + T.exp(-b_raw[0]))

                for jk, jv in T.Parallel(DK, block_DV):
                    h_fragment[jk, jv] *= exp_g[0]

                for jk, jv in T.Parallel(DK, block_DV):
                    if bv * block_DV + jv < DV:
                        reduce_fragment[jk, jv] = h_fragment[jk, jv] * k_fragment[jk]
                    else:
                        reduce_fragment[jk, jv] = 0.0
                T.reduce_sum(reduce_fragment, kv_fragment, dim=0, clear=True)

                for jv in T.Parallel(block_DV):
                    if bv * block_DV + jv < DV:
                        v_fragment[jv] = v[0, token_idx, bh, bv * block_DV + jv]
                        v_fragment[jv] = (v_fragment[jv] - kv_fragment[jv]) * beta[0]

                for jk, jv in T.Parallel(DK, block_DV):
                    if bv * block_DV + jv < DV:
                        h_fragment[jk, jv] += k_fragment[jk] * v_fragment[jv]

                for jk, jv in T.Parallel(DK, block_DV):
                    if bv * block_DV + jv < DV:
                        reduce_fragment[jk, jv] = h_fragment[jk, jv] * q_fragment[jk]
                    else:
                        reduce_fragment[jk, jv] = 0.0
                T.reduce_sum(reduce_fragment, o_fragment, dim=0, clear=True)

                for jv in T.Parallel(block_DV):
                    if bv * block_DV + jv < DV:
                        o[0, token_idx, bh, bv * block_DV + jv] = o_fragment[jv]

                if cache_intermediate_states:
                    if cache_idx >= 0:
                        for jk, jv in T.Parallel(DK, block_DV):
                            if bv * block_DV + jv < DV:
                                intermediate_states_buffer[
                                    cache_idx, step, bh, bv * block_DV + jv, jk
                                ] = h_fragment[jk, jv]
                    if has_tree_attention:
                        T.barrier_arrive(cache_barrier)
                        T.barrier_wait(cache_barrier, step % 2)

            if not disable_state_update:
                if state_idx >= 0:
                    for jk, jv in T.Parallel(DK, block_DV):
                        if bv * block_DV + jv < DV:
                            h0_source[state_idx, bh, bv * block_DV + jv, jk] = (
                                h_fragment[jk, jv]
                            )

    return tilelang_flashqla_gdn_regular_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_flashqla_gdn_regular_fullv(
    H,
    HV,
    DK,
    DV,
    tokens_per_seq,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    disable_state_update,
    cache_intermediate_states,
    has_tree_attention,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    cache_steps = T.dynamic("cache_steps")

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)
    intermediate_shape = (num_sequences, cache_steps, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_regular_fullv_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        intermediate_states_buffer: T.Tensor(intermediate_shape, dtype=state_dtype),
        intermediate_state_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        retrieve_parent_token: T.Tensor((num_sequences, cache_steps), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV, threads=128) as (bid,):
            bn = bid // HV
            bh = bid % HV
            bhq = bh // (HV // H)

            seq_start = T.alloc_var("int32")
            state_idx = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")

            seq_start = bn * tokens_per_seq
            state_idx = h0_indices[bn]
            cache_idx = -1
            if cache_intermediate_states:
                cache_idx = intermediate_state_indices[bn]

            h_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            q_fragment = T.alloc_fragment((DK,), dtype=accum_dtype)
            k_fragment = T.alloc_fragment((DK,), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((DV,), dtype=accum_dtype)
            kv_fragment = T.alloc_fragment((DV,), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((DV,), dtype=accum_dtype)
            reduce_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            q_norm = T.alloc_fragment((1,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((1,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            parent_step = T.alloc_fragment((1,), dtype=indices_dtype)
            cache_barrier = T.alloc_barrier(arrive_count=128)

            T.clear(h_fragment)
            if state_idx >= 0:
                for jk, jv in T.Parallel(DK, DV):
                    h_fragment[jk, jv] = h0_source[state_idx, bh, jv, jk]

            for step in T.serial(tokens_per_seq):
                if has_tree_attention:
                    if step != 0 and cache_idx >= 0:
                        parent_step[0] = retrieve_parent_token[bn, step]
                        for jk, jv in T.Parallel(DK, DV):
                            h_fragment[jk, jv] = intermediate_states_buffer[
                                cache_idx, parent_step[0], bh, jv, jk
                            ]

                token_idx = seq_start + step

                for jk in T.Parallel(DK):
                    q_fragment[jk] = q[0, token_idx, bhq, jk]
                    k_fragment[jk] = k[0, token_idx, bhq, jk]

                if use_qk_l2norm_in_kernel:
                    q_norm[0] = 0.0
                    k_norm[0] = 0.0
                    for jk in T.serial(DK):
                        q_norm[0] += q_fragment[jk] * q_fragment[jk]
                        k_norm[0] += k_fragment[jk] * k_fragment[jk]
                    q_norm[0] = T.rsqrt(q_norm[0] + 1e-6)
                    k_norm[0] = T.rsqrt(k_norm[0] + 1e-6)
                    for jk in T.Parallel(DK):
                        q_fragment[jk] *= q_norm[0]
                        k_fragment[jk] *= k_norm[0]

                for jk in T.Parallel(DK):
                    q_fragment[jk] *= scale

                a_raw[0] = a[0, token_idx, bh]
                b_raw[0] = b[0, token_idx, bh]
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = T.log(1.0 + T.exp(beta_x[0])) / softplus_beta
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp(-T.exp(A_log[bh]) * softplus_x[0])
                beta[0] = 1.0 / (1.0 + T.exp(-b_raw[0]))

                for jk, jv in T.Parallel(DK, DV):
                    h_fragment[jk, jv] *= exp_g[0]

                for jk, jv in T.Parallel(DK, DV):
                    reduce_fragment[jk, jv] = h_fragment[jk, jv] * k_fragment[jk]
                T.reduce_sum(reduce_fragment, kv_fragment, dim=0, clear=True)

                for jv in T.Parallel(DV):
                    v_fragment[jv] = v[0, token_idx, bh, jv]
                    v_fragment[jv] = (v_fragment[jv] - kv_fragment[jv]) * beta[0]

                for jk, jv in T.Parallel(DK, DV):
                    h_fragment[jk, jv] += k_fragment[jk] * v_fragment[jv]

                for jk, jv in T.Parallel(DK, DV):
                    reduce_fragment[jk, jv] = h_fragment[jk, jv] * q_fragment[jk]
                T.reduce_sum(reduce_fragment, o_fragment, dim=0, clear=True)

                for jv in T.Parallel(DV):
                    o[0, token_idx, bh, jv] = o_fragment[jv]

                if cache_intermediate_states:
                    if cache_idx >= 0:
                        for jk, jv in T.Parallel(DK, DV):
                            intermediate_states_buffer[cache_idx, step, bh, jv, jk] = (
                                h_fragment[jk, jv]
                            )
                    if has_tree_attention:
                        T.barrier_arrive(cache_barrier)
                        T.barrier_wait(cache_barrier, step % 2)

            if not disable_state_update:
                if state_idx >= 0:
                    for jk, jv in T.Parallel(DK, DV):
                        h0_source[state_idx, bh, jv, jk] = h_fragment[jk, jv]

    return tilelang_flashqla_gdn_regular_fullv_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_flashqla_gdn_decode_fullv(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
    max_nreg: int = 0,
    identity_indices: bool = False,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_fullv_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV, threads=128) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bn = bid // HV
            bh = bid % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            if identity_indices:
                state_idx = bn
            else:
                state_idx = h0_indices[bn]

            h_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            q_fragment = T.alloc_fragment((DK,), dtype=accum_dtype)
            k_fragment = T.alloc_fragment((DK,), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((DV,), dtype=accum_dtype)
            kv_fragment = T.alloc_fragment((DV,), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((DV,), dtype=accum_dtype)
            reduce_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            q_norm = T.alloc_fragment((1,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((1,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)

            T.clear(h_fragment)
            if state_idx >= 0:
                for jk, jv in T.Parallel(DK, DV):
                    h_fragment[jk, jv] = h0_source[state_idx, bh, jv, jk]

            for jk in T.Parallel(DK):
                q_fragment[jk] = q[0, bn, bhq, jk]
                k_fragment[jk] = k[0, bn, bhq, jk]

            if use_qk_l2norm_in_kernel:
                q_norm[0] = 0.0
                k_norm[0] = 0.0
                for jk in T.serial(DK):
                    q_norm[0] += q_fragment[jk] * q_fragment[jk]
                    k_norm[0] += k_fragment[jk] * k_fragment[jk]
                q_norm[0] = T.rsqrt(q_norm[0] + 1e-6)
                k_norm[0] = T.rsqrt(k_norm[0] + 1e-6)
                for jk in T.Parallel(DK):
                    q_fragment[jk] *= q_norm[0]
                    k_fragment[jk] *= k_norm[0]

            for jk in T.Parallel(DK):
                q_fragment[jk] *= scale

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            if a_is_log_decay:
                exp_g[0] = T.exp2(a_raw[0] * 1.442695)
            else:
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = (
                        T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                        / softplus_beta
                    )
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp2(
                    -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
                )
            if b_is_beta:
                beta[0] = b_raw[0]
            else:
                beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for jk, jv in T.Parallel(DK, DV):
                h_fragment[jk, jv] *= exp_g[0]

            for jk, jv in T.Parallel(DK, DV):
                reduce_fragment[jk, jv] = h_fragment[jk, jv] * k_fragment[jk]
            T.reduce_sum(reduce_fragment, kv_fragment, dim=0, clear=True)

            for jv in T.Parallel(DV):
                v_fragment[jv] = v[0, bn, bh, jv]
                v_fragment[jv] = (v_fragment[jv] - kv_fragment[jv]) * beta[0]

            for jk, jv in T.Parallel(DK, DV):
                h_fragment[jk, jv] += k_fragment[jk] * v_fragment[jv]

            for jk, jv in T.Parallel(DK, DV):
                reduce_fragment[jk, jv] = h_fragment[jk, jv] * q_fragment[jk]
            T.reduce_sum(reduce_fragment, o_fragment, dim=0, clear=True)

            for jv in T.Parallel(DV):
                o[0, bn, bh, jv] = o_fragment[jv]

            if state_idx >= 0:
                for jk, jv in T.Parallel(DK, DV):
                    h0_source[state_idx, bh, jv, jk] = h_fragment[jk, jv]

    return tilelang_flashqla_gdn_decode_fullv_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_flashqla_gdn_decode_bv32_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    block_DV = 32
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_bv32_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=32) as (bid,):
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            state_idx = h0_indices[bn]

            # DK is fixed to 128 by the Python wrapper, so each warp lane owns four K values.
            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_new_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_new_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_new_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_new_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            if a_is_log_decay:
                exp_g[0] = T.exp2(a_raw[0] * 1.442695)
            else:
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = (
                        T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                        / softplus_beta
                    )
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp2(
                    -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
                )
            if b_is_beta:
                beta[0] = b_raw[0]
            else:
                beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx]
                q_lane_1[tx] = q[0, bn, bhq, tx + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )

                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]

                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

                for jv in T.serial(block_DV):
                    h_lane_0[tx] = 0.0
                    h_lane_1[tx] = 0.0
                    h_lane_2[tx] = 0.0
                    h_lane_3[tx] = 0.0
                    if state_idx >= 0:
                        h_lane_0[tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx
                        ]
                        h_lane_1[tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 32
                        ]
                        h_lane_2[tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 64
                        ]
                        h_lane_3[tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 96
                        ]
                    h_lane_0[tx] *= exp_g[0]
                    h_lane_1[tx] *= exp_g[0]
                    h_lane_2[tx] *= exp_g[0]
                    h_lane_3[tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )

                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, bv * block_DV + jv] - kv_value[tx]
                    ) * beta[0]

                    h_new_lane_0[tx] = h_lane_0[tx] + k_lane_0[tx] * v_delta[tx]
                    h_new_lane_1[tx] = h_lane_1[tx] + k_lane_1[tx] * v_delta[tx]
                    h_new_lane_2[tx] = h_lane_2[tx] + k_lane_2[tx] * v_delta[tx]
                    h_new_lane_3[tx] = h_lane_3[tx] + k_lane_3[tx] * v_delta[tx]
                    if state_idx >= 0:
                        h0_source[state_idx, bh, bv * block_DV + jv, tx] = (
                            h_new_lane_0[tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 32] = (
                            h_new_lane_1[tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 64] = (
                            h_new_lane_2[tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 96] = (
                            h_new_lane_3[tx]
                        )
                    local_sum[tx] = (
                        h_new_lane_0[tx] * q_lane_0[tx]
                        + h_new_lane_1[tx] * q_lane_1[tx]
                        + h_new_lane_2[tx] * q_lane_2[tx]
                        + h_new_lane_3[tx] * q_lane_3[tx]
                    )

                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if tx == 0:
                        o[0, bn, bh, bv * block_DV + jv] = o_value[tx]

    return tilelang_flashqla_gdn_decode_bv32_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_flashqla_gdn_decode_bv16_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    block_DV = 16
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_bv16_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=32) as (bid,):
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            state_idx = h0_indices[bn]

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_new_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_new_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_new_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_new_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            x[0] = a_raw[0] + dt_bias[bh]
            beta_x[0] = softplus_beta * x[0]
            if beta_x[0] <= softplus_threshold:
                softplus_x[0] = (
                    T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                    / softplus_beta
                )
            else:
                softplus_x[0] = x[0]
            exp_g[0] = T.exp2(
                -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
            )
            beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx]
                q_lane_1[tx] = q[0, bn, bhq, tx + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )

                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]

                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

                for jv in T.serial(block_DV):
                    h_lane_0[tx] = 0.0
                    h_lane_1[tx] = 0.0
                    h_lane_2[tx] = 0.0
                    h_lane_3[tx] = 0.0
                    if state_idx >= 0:
                        h_lane_0[tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx
                        ]
                        h_lane_1[tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 32
                        ]
                        h_lane_2[tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 64
                        ]
                        h_lane_3[tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 96
                        ]
                    h_lane_0[tx] *= exp_g[0]
                    h_lane_1[tx] *= exp_g[0]
                    h_lane_2[tx] *= exp_g[0]
                    h_lane_3[tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )

                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, bv * block_DV + jv] - kv_value[tx]
                    ) * beta[0]

                    h_new_lane_0[tx] = h_lane_0[tx] + k_lane_0[tx] * v_delta[tx]
                    h_new_lane_1[tx] = h_lane_1[tx] + k_lane_1[tx] * v_delta[tx]
                    h_new_lane_2[tx] = h_lane_2[tx] + k_lane_2[tx] * v_delta[tx]
                    h_new_lane_3[tx] = h_lane_3[tx] + k_lane_3[tx] * v_delta[tx]
                    if state_idx >= 0:
                        h0_source[state_idx, bh, bv * block_DV + jv, tx] = (
                            h_new_lane_0[tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 32] = (
                            h_new_lane_1[tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 64] = (
                            h_new_lane_2[tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 96] = (
                            h_new_lane_3[tx]
                        )
                    local_sum[tx] = (
                        h_new_lane_0[tx] * q_lane_0[tx]
                        + h_new_lane_1[tx] * q_lane_1[tx]
                        + h_new_lane_2[tx] * q_lane_2[tx]
                        + h_new_lane_3[tx] * q_lane_3[tx]
                    )

                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if tx == 0:
                        o[0, bn, bh, bv * block_DV + jv] = o_value[tx]

    return tilelang_flashqla_gdn_decode_bv16_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_bv32x2_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    block_DV = 32
    value_span = block_DV * 2
    num_value_tile_groups = tilelang.cdiv(DV, value_span)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_bv32x2_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tile_groups, threads=64) as (bid,):
            bnh = bid // num_value_tile_groups
            bv = bid % num_value_tile_groups
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            value_offset = T.alloc_var("int32")
            state_idx = h0_indices[bn]

            q_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_new_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_new_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_new_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_new_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((64,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((64,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((64,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((64,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((64,), dtype=accum_dtype)
            o_value = T.alloc_fragment((64,), dtype=accum_dtype)

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            x[0] = a_raw[0] + dt_bias[bh]
            beta_x[0] = softplus_beta * x[0]
            if beta_x[0] <= softplus_threshold:
                softplus_x[0] = (
                    T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                    / softplus_beta
                )
            else:
                softplus_x[0] = x[0]
            exp_g[0] = T.exp2(
                -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
            )
            beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for tx in T.Parallel(64):
                q_lane_0[tx] = q[0, bn, bhq, tx % 32]
                q_lane_1[tx] = q[0, bn, bhq, tx % 32 + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx % 32 + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx % 32 + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx % 32]
                k_lane_1[tx] = k[0, bn, bhq, tx % 32 + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx % 32 + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx % 32 + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )

                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]

                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

                for jv in T.serial(block_DV):
                    value_offset = bv * value_span + (tx // 32) * block_DV + jv
                    h_lane_0[tx] = 0.0
                    h_lane_1[tx] = 0.0
                    h_lane_2[tx] = 0.0
                    h_lane_3[tx] = 0.0
                    if state_idx >= 0:
                        if value_offset < DV:
                            h_lane_0[tx] = h0_source[state_idx, bh, value_offset, tx % 32]
                            h_lane_1[tx] = h0_source[
                                state_idx, bh, value_offset, tx % 32 + 32
                            ]
                            h_lane_2[tx] = h0_source[
                                state_idx, bh, value_offset, tx % 32 + 64
                            ]
                            h_lane_3[tx] = h0_source[
                                state_idx, bh, value_offset, tx % 32 + 96
                            ]
                    h_lane_0[tx] *= exp_g[0]
                    h_lane_1[tx] *= exp_g[0]
                    h_lane_2[tx] *= exp_g[0]
                    h_lane_3[tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )

                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if value_offset < DV:
                        v_delta[tx] = (v[0, bn, bh, value_offset] - kv_value[tx]) * beta[0]

                        h_new_lane_0[tx] = h_lane_0[tx] + k_lane_0[tx] * v_delta[tx]
                        h_new_lane_1[tx] = h_lane_1[tx] + k_lane_1[tx] * v_delta[tx]
                        h_new_lane_2[tx] = h_lane_2[tx] + k_lane_2[tx] * v_delta[tx]
                        h_new_lane_3[tx] = h_lane_3[tx] + k_lane_3[tx] * v_delta[tx]
                        if state_idx >= 0:
                            h0_source[state_idx, bh, value_offset, tx % 32] = h_new_lane_0[tx]
                            h0_source[state_idx, bh, value_offset, tx % 32 + 32] = (
                                h_new_lane_1[tx]
                            )
                            h0_source[state_idx, bh, value_offset, tx % 32 + 64] = (
                                h_new_lane_2[tx]
                            )
                            h0_source[state_idx, bh, value_offset, tx % 32 + 96] = (
                                h_new_lane_3[tx]
                            )
                        local_sum[tx] = (
                            h_new_lane_0[tx] * q_lane_0[tx]
                            + h_new_lane_1[tx] * q_lane_1[tx]
                            + h_new_lane_2[tx] * q_lane_2[tx]
                            + h_new_lane_3[tx] * q_lane_3[tx]
                        )

                        o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                        if tx % 32 == 0:
                            o[0, bn, bh, value_offset] = o_value[tx]

    return tilelang_flashqla_gdn_decode_bv32x2_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_bv16x2_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    identity_indices: bool = False,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    block_DV = 16
    value_span = block_DV * 2
    num_value_tile_groups = tilelang.cdiv(DV, value_span)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_bv16x2_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tile_groups, threads=64) as (bid,):
            bnh = bid // num_value_tile_groups
            bv = bid % num_value_tile_groups
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            value_offset = T.alloc_var("int32")
            if identity_indices:
                state_idx = bn
            else:
                state_idx = h0_indices[bn]

            q_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((64,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((64,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((64,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((64,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((64,), dtype=accum_dtype)
            o_value = T.alloc_fragment((64,), dtype=accum_dtype)

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            x[0] = a_raw[0] + dt_bias[bh]
            beta_x[0] = softplus_beta * x[0]
            if beta_x[0] <= softplus_threshold:
                softplus_x[0] = (
                    T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                    / softplus_beta
                )
            else:
                softplus_x[0] = x[0]
            exp_g[0] = T.exp2(
                -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
            )
            beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for tx in T.Parallel(64):
                q_lane_0[tx] = q[0, bn, bhq, tx % 32]
                q_lane_1[tx] = q[0, bn, bhq, tx % 32 + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx % 32 + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx % 32 + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx % 32]
                k_lane_1[tx] = k[0, bn, bhq, tx % 32 + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx % 32 + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx % 32 + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]
                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

                for jv in T.serial(block_DV):
                    value_offset = bv * value_span + (tx // 32) * block_DV + jv
                    h_lane_0[tx] = 0.0
                    h_lane_1[tx] = 0.0
                    h_lane_2[tx] = 0.0
                    h_lane_3[tx] = 0.0
                    if value_offset < DV:
                        h_lane_0[tx] = h0_source[state_idx, bh, value_offset, tx % 32]
                        h_lane_1[tx] = h0_source[
                            state_idx, bh, value_offset, tx % 32 + 32
                        ]
                        h_lane_2[tx] = h0_source[
                            state_idx, bh, value_offset, tx % 32 + 64
                        ]
                        h_lane_3[tx] = h0_source[
                            state_idx, bh, value_offset, tx % 32 + 96
                        ]
                    h_lane_0[tx] *= exp_g[0]
                    h_lane_1[tx] *= exp_g[0]
                    h_lane_2[tx] *= exp_g[0]
                    h_lane_3[tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if value_offset < DV:
                        v_delta[tx] = (v[0, bn, bh, value_offset] - kv_value[tx]) * beta[0]
                        h_lane_0[tx] += k_lane_0[tx] * v_delta[tx]
                        h_lane_1[tx] += k_lane_1[tx] * v_delta[tx]
                        h_lane_2[tx] += k_lane_2[tx] * v_delta[tx]
                        h_lane_3[tx] += k_lane_3[tx] * v_delta[tx]
                        h0_source[state_idx, bh, value_offset, tx % 32] = h_lane_0[tx]
                        h0_source[state_idx, bh, value_offset, tx % 32 + 32] = (
                            h_lane_1[tx]
                        )
                        h0_source[state_idx, bh, value_offset, tx % 32 + 64] = (
                            h_lane_2[tx]
                        )
                        h0_source[state_idx, bh, value_offset, tx % 32 + 96] = (
                            h_lane_3[tx]
                        )
                        local_sum[tx] = (
                            h_lane_0[tx] * q_lane_0[tx]
                            + h_lane_1[tx] * q_lane_1[tx]
                            + h_lane_2[tx] * q_lane_2[tx]
                            + h_lane_3[tx] * q_lane_3[tx]
                        )
                        o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                        if tx % 32 == 0:
                            o[0, bn, bh, value_offset] = o_value[tx]

    return tilelang_flashqla_gdn_decode_bv16x2_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_flashqla_gdn_decode_prepare_gates(
    HV,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    gate_dtype,
    softplus_beta,
    softplus_threshold,
):
    total_tokens = T.dynamic("total_tokens")
    a_shape = (1, total_tokens, HV)
    gate_shape = (1, total_tokens, HV)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_prepare_gates_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        b: T.Tensor(a_shape, dtype=b_dtype),
        exp_g: T.Tensor(gate_shape, dtype=gate_dtype),
        beta: T.Tensor(gate_shape, dtype=gate_dtype),
    ):
        with T.Kernel(total_tokens * HV, threads=128) as (bid,):
            token = bid // HV
            bh = bid % HV
            a_raw = T.alloc_fragment((1,), dtype=gate_dtype)
            b_raw = T.alloc_fragment((1,), dtype=gate_dtype)
            x = T.alloc_fragment((1,), dtype=gate_dtype)
            beta_x = T.alloc_fragment((1,), dtype=gate_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=gate_dtype)

            a_raw[0] = a[0, token, bh]
            b_raw[0] = b[0, token, bh]
            x[0] = a_raw[0] + dt_bias[bh]
            beta_x[0] = softplus_beta * x[0]
            if beta_x[0] <= softplus_threshold:
                softplus_x[0] = T.log(1.0 + T.exp(beta_x[0])) / softplus_beta
            else:
                softplus_x[0] = x[0]
            exp_g[0, token, bh] = T.exp(-T.exp(A_log[bh]) * softplus_x[0])
            beta[0, token, bh] = 1.0 / (1.0 + T.exp(-b_raw[0]))

    return tilelang_flashqla_gdn_decode_prepare_gates_kernel


def _prepare_decode_gates(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
):
    exp_g = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(a, dtype=torch.float32)
    kernel = _get_recurrent_kernel(
        tilelang_flashqla_gdn_decode_prepare_gates,
        HV=a.shape[-1],
        a_dtype=a.dtype,
        b_dtype=b.dtype,
        a_log_dtype=A_log.dtype,
        dt_bias_dtype=dt_bias.dtype,
        gate_dtype=torch.float32,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )
    kernel(A_log.contiguous(), a, dt_bias.contiguous(), b, exp_g, beta)
    return exp_g, beta


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_bv16_identity_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 16,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    num_sequences = T.dynamic("num_sequences")
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, num_sequences, H, DK)
    k_shape = (1, num_sequences, H, DK)
    v_shape = (1, num_sequences, HV, DV)
    a_shape = (1, num_sequences, HV)
    b_shape = (1, num_sequences, HV)
    o_shape = (1, num_sequences, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_bv16_identity_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=32) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            if a_is_log_decay:
                exp_g[0] = T.exp2(a_raw[0] * 1.442695)
            else:
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = (
                        T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                        / softplus_beta
                    )
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp2(
                    -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
                )
            if b_is_beta:
                beta[0] = b_raw[0]
            else:
                beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h_lane_0[jv, tx] = h0_source[
                        bn, bh, bv * block_DV + jv, tx
                    ]
                    h_lane_1[jv, tx] = h0_source[
                        bn, bh, bv * block_DV + jv, tx + 32
                    ]
                    h_lane_2[jv, tx] = h0_source[
                        bn, bh, bv * block_DV + jv, tx + 64
                    ]
                    h_lane_3[jv, tx] = h0_source[
                        bn, bh, bv * block_DV + jv, tx + 96
                    ]

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx]
                q_lane_1[tx] = q[0, bn, bhq, tx + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]
                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h_lane_0[jv, tx] *= exp_g[0]
                    h_lane_1[jv, tx] *= exp_g[0]
                    h_lane_2[jv, tx] *= exp_g[0]
                    h_lane_3[jv, tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * k_lane_0[tx]
                        + h_lane_1[jv, tx] * k_lane_1[tx]
                        + h_lane_2[jv, tx] * k_lane_2[tx]
                        + h_lane_3[jv, tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, bv * block_DV + jv] - kv_value[tx]
                    ) * beta[0]
                    h_lane_0[jv, tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[jv, tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[jv, tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[jv, tx] += k_lane_3[tx] * v_delta[tx]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * q_lane_0[tx]
                        + h_lane_1[jv, tx] * q_lane_1[tx]
                        + h_lane_2[jv, tx] * q_lane_2[tx]
                        + h_lane_3[jv, tx] * q_lane_3[tx]
                    )
                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if tx == 0:
                        o[0, bn, bh, bv * block_DV + jv] = o_value[tx]

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h0_source[bn, bh, bv * block_DV + jv, tx] = h_lane_0[jv, tx]
                    h0_source[bn, bh, bv * block_DV + jv, tx + 32] = (
                        h_lane_1[jv, tx]
                    )
                    h0_source[bn, bh, bv * block_DV + jv, tx + 64] = (
                        h_lane_2[jv, tx]
                    )
                    h0_source[bn, bh, bv * block_DV + jv, tx + 96] = (
                        h_lane_3[jv, tx]
                    )

    return tilelang_flashqla_gdn_decode_bv16_identity_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_b4_bv16_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 16,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
    static_B: int = 4,
):
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, static_B, H, DK)
    k_shape = (1, static_B, H, DK)
    v_shape = (1, static_B, HV, DV)
    a_shape = (1, static_B, HV)
    b_shape = (1, static_B, HV)
    o_shape = (1, static_B, HV, DV)
    state_shape = (static_B, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_b4_bv16_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(static_B * HV * num_value_tiles, threads=32) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)
            in_bounds = T.alloc_var("bool")

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            if a_is_log_decay:
                exp_g[0] = T.exp2(a_raw[0] * 1.442695)
            else:
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = (
                        T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                        / softplus_beta
                    )
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp2(
                    -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
                )
            if b_is_beta:
                beta[0] = b_raw[0]
            else:
                beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    in_bounds = bv * block_DV + jv < DV
                    if in_bounds:
                        h_lane_0[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx
                        ]
                        h_lane_1[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx + 32
                        ]
                        h_lane_2[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx + 64
                        ]
                        h_lane_3[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx + 96
                        ]
                    else:
                        h_lane_0[jv, tx] = 0.0
                        h_lane_1[jv, tx] = 0.0
                        h_lane_2[jv, tx] = 0.0
                        h_lane_3[jv, tx] = 0.0

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx]
                q_lane_1[tx] = q[0, bn, bhq, tx + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]

                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    in_bounds = bv * block_DV + jv < DV
                    h_lane_0[jv, tx] *= exp_g[0]
                    h_lane_1[jv, tx] *= exp_g[0]
                    h_lane_2[jv, tx] *= exp_g[0]
                    h_lane_3[jv, tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * k_lane_0[tx]
                        + h_lane_1[jv, tx] * k_lane_1[tx]
                        + h_lane_2[jv, tx] * k_lane_2[tx]
                        + h_lane_3[jv, tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if in_bounds:
                        v_delta[tx] = (
                            v[0, bn, bh, bv * block_DV + jv] - kv_value[tx]
                        ) * beta[0]
                        h_lane_0[jv, tx] += k_lane_0[tx] * v_delta[tx]
                        h_lane_1[jv, tx] += k_lane_1[tx] * v_delta[tx]
                        h_lane_2[jv, tx] += k_lane_2[tx] * v_delta[tx]
                        h_lane_3[jv, tx] += k_lane_3[tx] * v_delta[tx]
                        local_sum[tx] = (
                            h_lane_0[jv, tx] * q_lane_0[tx]
                            + h_lane_1[jv, tx] * q_lane_1[tx]
                            + h_lane_2[jv, tx] * q_lane_2[tx]
                            + h_lane_3[jv, tx] * q_lane_3[tx]
                        )
                        o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                        if tx == 0:
                            o[0, bn, bh, bv * block_DV + jv] = o_value[tx]

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    if bv * block_DV + jv < DV:
                        h0_source[bn, bh, bv * block_DV + jv, tx] = h_lane_0[jv, tx]
                        h0_source[bn, bh, bv * block_DV + jv, tx + 32] = (
                            h_lane_1[jv, tx]
                        )
                        h0_source[bn, bh, bv * block_DV + jv, tx + 64] = (
                            h_lane_2[jv, tx]
                        )
                        h0_source[bn, bh, bv * block_DV + jv, tx + 96] = (
                            h_lane_3[jv, tx]
                        )

    return tilelang_flashqla_gdn_decode_b4_bv16_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_b4_precomputed_bv16_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    q_dtype,
    k_dtype,
    v_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    o_dtype,
    accum_dtype,
    max_nreg: int = 0,
):
    block_DV = 16
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, 4, H, DK)
    k_shape = (1, 4, H, DK)
    v_shape = (1, 4, HV, DV)
    g_shape = (1, 4, HV)
    beta_shape = (1, 4, HV)
    o_shape = (1, 4, HV, DV)
    state_shape = (4, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_b4_precomputed_bv16_warp_kernel(
        g: T.Tensor(g_shape, dtype=g_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        beta_in: T.Tensor(beta_shape, dtype=beta_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(4 * HV * num_value_tiles, threads=32) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)

            exp_g[0] = T.exp2(g[0, bn, bh] * 1.442695)
            beta[0] = beta_in[0, bn, bh]

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h_lane_0[jv, tx] = h0_source[bn, bh, bv * block_DV + jv, tx]
                    h_lane_1[jv, tx] = h0_source[
                        bn, bh, bv * block_DV + jv, tx + 32
                    ]
                    h_lane_2[jv, tx] = h0_source[
                        bn, bh, bv * block_DV + jv, tx + 64
                    ]
                    h_lane_3[jv, tx] = h0_source[
                        bn, bh, bv * block_DV + jv, tx + 96
                    ]

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx] * scale
                q_lane_1[tx] = q[0, bn, bhq, tx + 32] * scale
                q_lane_2[tx] = q[0, bn, bhq, tx + 64] * scale
                q_lane_3[tx] = q[0, bn, bhq, tx + 96] * scale
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h_lane_0[jv, tx] *= exp_g[0]
                    h_lane_1[jv, tx] *= exp_g[0]
                    h_lane_2[jv, tx] *= exp_g[0]
                    h_lane_3[jv, tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * k_lane_0[tx]
                        + h_lane_1[jv, tx] * k_lane_1[tx]
                        + h_lane_2[jv, tx] * k_lane_2[tx]
                        + h_lane_3[jv, tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, bv * block_DV + jv] - kv_value[tx]
                    ) * beta[0]
                    h_lane_0[jv, tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[jv, tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[jv, tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[jv, tx] += k_lane_3[tx] * v_delta[tx]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * q_lane_0[tx]
                        + h_lane_1[jv, tx] * q_lane_1[tx]
                        + h_lane_2[jv, tx] * q_lane_2[tx]
                        + h_lane_3[jv, tx] * q_lane_3[tx]
                    )
                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if tx == 0:
                        o[0, bn, bh, bv * block_DV + jv] = o_value[tx]

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h0_source[bn, bh, bv * block_DV + jv, tx] = h_lane_0[jv, tx]
                    h0_source[bn, bh, bv * block_DV + jv, tx + 32] = (
                        h_lane_1[jv, tx]
                    )
                    h0_source[bn, bh, bv * block_DV + jv, tx + 64] = (
                        h_lane_2[jv, tx]
                    )
                    h0_source[bn, bh, bv * block_DV + jv, tx + 96] = (
                        h_lane_3[jv, tx]
                    )

    return tilelang_flashqla_gdn_decode_b4_precomputed_bv16_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_precomputed_qknorm_bv16_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    q_dtype,
    k_dtype,
    v_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    o_dtype,
    accum_dtype,
    block_DV: int = 16,
    max_nreg: int = 0,
    static_B: int = 4,
):
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, static_B, H, DK)
    k_shape = (1, static_B, H, DK)
    v_shape = (1, static_B, HV, DV)
    g_shape = (1, static_B, HV)
    beta_shape = (1, static_B, HV)
    o_shape = (1, static_B, HV, DV)
    state_shape = (static_B, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_precomputed_qknorm_bv16_warp_kernel(
        g: T.Tensor(g_shape, dtype=g_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        beta_in: T.Tensor(beta_shape, dtype=beta_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(static_B * HV * num_value_tiles, threads=32) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)

            exp_g[0] = T.exp2(g[0, bn, bh] * 1.442695)
            beta[0] = beta_in[0, bn, bh]

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h_lane_0[jv, tx] = h0_source[bn, bh, bv * block_DV + jv, tx]
                    h_lane_1[jv, tx] = h0_source[
                        bn, bh, bv * block_DV + jv, tx + 32
                    ]
                    h_lane_2[jv, tx] = h0_source[
                        bn, bh, bv * block_DV + jv, tx + 64
                    ]
                    h_lane_3[jv, tx] = h0_source[
                        bn, bh, bv * block_DV + jv, tx + 96
                    ]

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx]
                q_lane_1[tx] = q[0, bn, bhq, tx + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                q_lane_0[tx] *= q_norm[tx] * scale
                q_lane_1[tx] *= q_norm[tx] * scale
                q_lane_2[tx] *= q_norm[tx] * scale
                q_lane_3[tx] *= q_norm[tx] * scale
                k_lane_0[tx] *= k_norm[tx]
                k_lane_1[tx] *= k_norm[tx]
                k_lane_2[tx] *= k_norm[tx]
                k_lane_3[tx] *= k_norm[tx]

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h_lane_0[jv, tx] *= exp_g[0]
                    h_lane_1[jv, tx] *= exp_g[0]
                    h_lane_2[jv, tx] *= exp_g[0]
                    h_lane_3[jv, tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * k_lane_0[tx]
                        + h_lane_1[jv, tx] * k_lane_1[tx]
                        + h_lane_2[jv, tx] * k_lane_2[tx]
                        + h_lane_3[jv, tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, bv * block_DV + jv] - kv_value[tx]
                    ) * beta[0]
                    h_lane_0[jv, tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[jv, tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[jv, tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[jv, tx] += k_lane_3[tx] * v_delta[tx]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * q_lane_0[tx]
                        + h_lane_1[jv, tx] * q_lane_1[tx]
                        + h_lane_2[jv, tx] * q_lane_2[tx]
                        + h_lane_3[jv, tx] * q_lane_3[tx]
                    )
                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if tx == 0:
                        o[0, bn, bh, bv * block_DV + jv] = o_value[tx]

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h0_source[bn, bh, bv * block_DV + jv, tx] = h_lane_0[jv, tx]
                    h0_source[bn, bh, bv * block_DV + jv, tx + 32] = (
                        h_lane_1[jv, tx]
                    )
                    h0_source[bn, bh, bv * block_DV + jv, tx + 64] = (
                        h_lane_2[jv, tx]
                    )
                    h0_source[bn, bh, bv * block_DV + jv, tx + 96] = (
                        h_lane_3[jv, tx]
                    )

    return tilelang_flashqla_gdn_decode_precomputed_qknorm_bv16_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_precomputed_qknorm_gqa3_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    q_dtype,
    k_dtype,
    v_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    o_dtype,
    accum_dtype,
    block_DV: int = 16,
    max_nreg: int = 0,
    static_B: int = 4,
):
    num_value_tiles = tilelang.cdiv(DV, block_DV)
    group_size = HV // H

    q_shape = (1, static_B, H, DK)
    k_shape = (1, static_B, H, DK)
    v_shape = (1, static_B, HV, DV)
    g_shape = (1, static_B, HV)
    beta_shape = (1, static_B, HV)
    o_shape = (1, static_B, HV, DV)
    state_shape = (static_B, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_precomputed_qknorm_gqa3_warp_kernel(
        g: T.Tensor(g_shape, dtype=g_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        beta_in: T.Tensor(beta_shape, dtype=beta_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(static_B * H * num_value_tiles, threads=96) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnk = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnk // H
            bhq = bnk % H

            q_lane_0 = T.alloc_fragment((96,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((96,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((96,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((96,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((96,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((96,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((96,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((96,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((96,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((96,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((96,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((96,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((96,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((96,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((96,), dtype=accum_dtype)
            beta = T.alloc_fragment((96,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((96,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((96,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((96,), dtype=accum_dtype)
            o_value = T.alloc_fragment((96,), dtype=accum_dtype)

            for t in T.Parallel(96):
                ig = t // 32
                lane = t % 32
                bh = bhq * group_size + ig
                q_lane_0[t] = q[0, bn, bhq, lane]
                q_lane_1[t] = q[0, bn, bhq, lane + 32]
                q_lane_2[t] = q[0, bn, bhq, lane + 64]
                q_lane_3[t] = q[0, bn, bhq, lane + 96]
                k_lane_0[t] = k[0, bn, bhq, lane]
                k_lane_1[t] = k[0, bn, bhq, lane + 32]
                k_lane_2[t] = k[0, bn, bhq, lane + 64]
                k_lane_3[t] = k[0, bn, bhq, lane + 96]
                q_norm[t] = (
                    q_lane_0[t] * q_lane_0[t]
                    + q_lane_1[t] * q_lane_1[t]
                    + q_lane_2[t] * q_lane_2[t]
                    + q_lane_3[t] * q_lane_3[t]
                )
                k_norm[t] = (
                    k_lane_0[t] * k_lane_0[t]
                    + k_lane_1[t] * k_lane_1[t]
                    + k_lane_2[t] * k_lane_2[t]
                    + k_lane_3[t] * k_lane_3[t]
                )
                q_norm[t] = T.warp_reduce_sum(q_norm[t])
                k_norm[t] = T.warp_reduce_sum(k_norm[t])
                q_norm[t] = T.rsqrt(q_norm[t] + 1e-6)
                k_norm[t] = T.rsqrt(k_norm[t] + 1e-6)
                q_lane_0[t] *= q_norm[t] * scale
                q_lane_1[t] *= q_norm[t] * scale
                q_lane_2[t] *= q_norm[t] * scale
                q_lane_3[t] *= q_norm[t] * scale
                k_lane_0[t] *= k_norm[t]
                k_lane_1[t] *= k_norm[t]
                k_lane_2[t] *= k_norm[t]
                k_lane_3[t] *= k_norm[t]
                exp_g[t] = T.exp2(g[0, bn, bh] * 1.442695)
                beta[t] = beta_in[0, bn, bh]

            for jv in T.serial(block_DV):
                for t in T.Parallel(96):
                    ig = t // 32
                    lane = t % 32
                    bh = bhq * group_size + ig
                    value_offset = bv * block_DV + jv
                    h_lane_0[t] = (
                        h0_source[bn, bh, value_offset, lane] * exp_g[t]
                    )
                    h_lane_1[t] = (
                        h0_source[bn, bh, value_offset, lane + 32] * exp_g[t]
                    )
                    h_lane_2[t] = (
                        h0_source[bn, bh, value_offset, lane + 64] * exp_g[t]
                    )
                    h_lane_3[t] = (
                        h0_source[bn, bh, value_offset, lane + 96] * exp_g[t]
                    )
                    local_sum[t] = (
                        h_lane_0[t] * k_lane_0[t]
                        + h_lane_1[t] * k_lane_1[t]
                        + h_lane_2[t] * k_lane_2[t]
                        + h_lane_3[t] * k_lane_3[t]
                    )
                    kv_value[t] = T.warp_reduce_sum(local_sum[t])
                    v_delta[t] = (
                        v[0, bn, bh, value_offset] - kv_value[t]
                    ) * beta[t]
                    h_lane_0[t] += k_lane_0[t] * v_delta[t]
                    h_lane_1[t] += k_lane_1[t] * v_delta[t]
                    h_lane_2[t] += k_lane_2[t] * v_delta[t]
                    h_lane_3[t] += k_lane_3[t] * v_delta[t]
                    local_sum[t] = (
                        h_lane_0[t] * q_lane_0[t]
                        + h_lane_1[t] * q_lane_1[t]
                        + h_lane_2[t] * q_lane_2[t]
                        + h_lane_3[t] * q_lane_3[t]
                    )
                    o_value[t] = T.warp_reduce_sum(local_sum[t])
                    if lane == 0:
                        o[0, bn, bh, value_offset] = o_value[t]
                    h0_source[bn, bh, value_offset, lane] = h_lane_0[t]
                    h0_source[bn, bh, value_offset, lane + 32] = h_lane_1[t]
                    h0_source[bn, bh, value_offset, lane + 64] = h_lane_2[t]
                    h0_source[bn, bh, value_offset, lane + 96] = h_lane_3[t]

    return tilelang_flashqla_gdn_decode_precomputed_qknorm_gqa3_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_b4_precomputed_stream_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    q_dtype,
    k_dtype,
    v_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    o_dtype,
    accum_dtype,
    block_DV: int = 16,
    max_nreg: int = 0,
):
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, 4, H, DK)
    k_shape = (1, 4, H, DK)
    v_shape = (1, 4, HV, DV)
    g_shape = (1, 4, HV)
    beta_shape = (1, 4, HV)
    o_shape = (1, 4, HV, DV)
    state_shape = (4, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_b4_precomputed_stream_warp_kernel(
        g: T.Tensor(g_shape, dtype=g_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        beta_in: T.Tensor(beta_shape, dtype=beta_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(4 * HV * num_value_tiles, threads=32) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)
            value_offset = T.alloc_var("int32")

            exp_g[0] = T.exp2(g[0, bn, bh] * 1.442695)
            beta[0] = beta_in[0, bn, bh]

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx] * scale
                q_lane_1[tx] = q[0, bn, bhq, tx + 32] * scale
                q_lane_2[tx] = q[0, bn, bhq, tx + 64] * scale
                q_lane_3[tx] = q[0, bn, bhq, tx + 96] * scale
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]

            for jv in T.serial(block_DV):
                value_offset = bv * block_DV + jv
                for tx in T.Parallel(32):
                    h_lane_0[tx] = h0_source[bn, bh, value_offset, tx] * exp_g[0]
                    h_lane_1[tx] = (
                        h0_source[bn, bh, value_offset, tx + 32] * exp_g[0]
                    )
                    h_lane_2[tx] = (
                        h0_source[bn, bh, value_offset, tx + 64] * exp_g[0]
                    )
                    h_lane_3[tx] = (
                        h0_source[bn, bh, value_offset, tx + 96] * exp_g[0]
                    )
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, value_offset] - kv_value[tx]
                    ) * beta[0]
                    h_lane_0[tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[tx] += k_lane_3[tx] * v_delta[tx]
                    local_sum[tx] = (
                        h_lane_0[tx] * q_lane_0[tx]
                        + h_lane_1[tx] * q_lane_1[tx]
                        + h_lane_2[tx] * q_lane_2[tx]
                        + h_lane_3[tx] * q_lane_3[tx]
                    )
                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if tx == 0:
                        o[0, bn, bh, value_offset] = o_value[tx]
                    h0_source[bn, bh, value_offset, tx] = h_lane_0[tx]
                    h0_source[bn, bh, value_offset, tx + 32] = h_lane_1[tx]
                    h0_source[bn, bh, value_offset, tx + 64] = h_lane_2[tx]
                    h0_source[bn, bh, value_offset, tx + 96] = h_lane_3[tx]

    return tilelang_flashqla_gdn_decode_b4_precomputed_stream_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_b2_grouped_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 8,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, 2, H, DK)
    k_shape = (1, 2, H, DK)
    v_shape = (1, 2, HV, DV)
    a_shape = (1, 2, HV)
    b_shape = (1, 2, HV)
    o_shape = (1, 2, HV, DV)
    state_shape = (2, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_b2_grouped_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(HV * num_value_tiles, threads=32) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bhq = bh // (HV // H)

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)
            value_offset = T.alloc_var("int32")

            for bn in T.serial(2):
                a_raw[0] = a[0, bn, bh]
                b_raw[0] = b[0, bn, bh]
                if a_is_log_decay:
                    exp_g[0] = T.exp2(a_raw[0] * 1.442695)
                else:
                    x[0] = a_raw[0] + dt_bias[bh]
                    beta_x[0] = softplus_beta * x[0]
                    if beta_x[0] <= softplus_threshold:
                        softplus_x[0] = (
                            T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                            / softplus_beta
                        )
                    else:
                        softplus_x[0] = x[0]
                    exp_g[0] = T.exp2(
                        -T.exp2(A_log[bh] * 1.442695)
                        * softplus_x[0]
                        * 1.442695
                    )
                if b_is_beta:
                    beta[0] = b_raw[0]
                else:
                    beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

                for tx in T.Parallel(32):
                    q_lane_0[tx] = q[0, bn, bhq, tx]
                    q_lane_1[tx] = q[0, bn, bhq, tx + 32]
                    q_lane_2[tx] = q[0, bn, bhq, tx + 64]
                    q_lane_3[tx] = q[0, bn, bhq, tx + 96]
                    k_lane_0[tx] = k[0, bn, bhq, tx]
                    k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                    k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                    k_lane_3[tx] = k[0, bn, bhq, tx + 96]
                    q_norm[tx] = (
                        q_lane_0[tx] * q_lane_0[tx]
                        + q_lane_1[tx] * q_lane_1[tx]
                        + q_lane_2[tx] * q_lane_2[tx]
                        + q_lane_3[tx] * q_lane_3[tx]
                    )
                    k_norm[tx] = (
                        k_lane_0[tx] * k_lane_0[tx]
                        + k_lane_1[tx] * k_lane_1[tx]
                        + k_lane_2[tx] * k_lane_2[tx]
                        + k_lane_3[tx] * k_lane_3[tx]
                    )
                    if use_qk_l2norm_in_kernel:
                        q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                        k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                        q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                        k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                        q_lane_0[tx] *= q_norm[tx]
                        q_lane_1[tx] *= q_norm[tx]
                        q_lane_2[tx] *= q_norm[tx]
                        q_lane_3[tx] *= q_norm[tx]
                        k_lane_0[tx] *= k_norm[tx]
                        k_lane_1[tx] *= k_norm[tx]
                        k_lane_2[tx] *= k_norm[tx]
                        k_lane_3[tx] *= k_norm[tx]
                    q_lane_0[tx] *= scale
                    q_lane_1[tx] *= scale
                    q_lane_2[tx] *= scale
                    q_lane_3[tx] *= scale

                for jv in T.serial(block_DV):
                    value_offset = bv * block_DV + jv
                    for tx in T.Parallel(32):
                        h_lane_0[tx] = (
                            h0_source[bn, bh, value_offset, tx] * exp_g[0]
                        )
                        h_lane_1[tx] = (
                            h0_source[bn, bh, value_offset, tx + 32]
                            * exp_g[0]
                        )
                        h_lane_2[tx] = (
                            h0_source[bn, bh, value_offset, tx + 64]
                            * exp_g[0]
                        )
                        h_lane_3[tx] = (
                            h0_source[bn, bh, value_offset, tx + 96]
                            * exp_g[0]
                        )
                        local_sum[tx] = (
                            h_lane_0[tx] * k_lane_0[tx]
                            + h_lane_1[tx] * k_lane_1[tx]
                            + h_lane_2[tx] * k_lane_2[tx]
                            + h_lane_3[tx] * k_lane_3[tx]
                        )
                        kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                        v_delta[tx] = (
                            v[0, bn, bh, value_offset] - kv_value[tx]
                        ) * beta[0]
                        h_lane_0[tx] += k_lane_0[tx] * v_delta[tx]
                        h_lane_1[tx] += k_lane_1[tx] * v_delta[tx]
                        h_lane_2[tx] += k_lane_2[tx] * v_delta[tx]
                        h_lane_3[tx] += k_lane_3[tx] * v_delta[tx]
                        local_sum[tx] = (
                            h_lane_0[tx] * q_lane_0[tx]
                            + h_lane_1[tx] * q_lane_1[tx]
                            + h_lane_2[tx] * q_lane_2[tx]
                            + h_lane_3[tx] * q_lane_3[tx]
                        )
                        o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                        if tx == 0:
                            o[0, bn, bh, value_offset] = o_value[tx]
                        h0_source[bn, bh, value_offset, tx] = h_lane_0[tx]
                        h0_source[bn, bh, value_offset, tx + 32] = h_lane_1[tx]
                        h0_source[bn, bh, value_offset, tx + 64] = h_lane_2[tx]
                        h0_source[bn, bh, value_offset, tx + 96] = h_lane_3[tx]

    return tilelang_flashqla_gdn_decode_b2_grouped_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_b2_dual_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 4,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, 2, H, DK)
    k_shape = (1, 2, H, DK)
    v_shape = (1, 2, HV, DV)
    a_shape = (1, 2, HV)
    b_shape = (1, 2, HV)
    o_shape = (1, 2, HV, DV)
    state_shape = (2, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_b2_dual_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(HV * num_value_tiles, threads=64) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bhq = bh // (HV // H)

            q_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((64,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((64,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((64,), dtype=accum_dtype)
            beta = T.alloc_fragment((64,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((64,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((64,), dtype=accum_dtype)
            x = T.alloc_fragment((64,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((64,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((64,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((64,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((64,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((64,), dtype=accum_dtype)
            o_value = T.alloc_fragment((64,), dtype=accum_dtype)
            bn = T.alloc_var("int32")
            lane = T.alloc_var("int32")
            value_offset = T.alloc_var("int32")

            for tx in T.Parallel(64):
                bn = tx // 32
                lane = tx % 32
                q_lane_0[tx] = q[0, bn, bhq, lane]
                q_lane_1[tx] = q[0, bn, bhq, lane + 32]
                q_lane_2[tx] = q[0, bn, bhq, lane + 64]
                q_lane_3[tx] = q[0, bn, bhq, lane + 96]
                k_lane_0[tx] = k[0, bn, bhq, lane]
                k_lane_1[tx] = k[0, bn, bhq, lane + 32]
                k_lane_2[tx] = k[0, bn, bhq, lane + 64]
                k_lane_3[tx] = k[0, bn, bhq, lane + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]
                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

                a_raw[tx] = a[0, bn, bh]
                b_raw[tx] = b[0, bn, bh]
                if a_is_log_decay:
                    exp_g[tx] = T.exp2(a_raw[tx] * 1.442695)
                else:
                    x[tx] = a_raw[tx] + dt_bias[bh]
                    beta_x[tx] = softplus_beta * x[tx]
                    if beta_x[tx] <= softplus_threshold:
                        softplus_x[tx] = (
                            T.log(1.0 + T.exp2(beta_x[tx] * 1.442695))
                            / softplus_beta
                        )
                    else:
                        softplus_x[tx] = x[tx]
                    exp_g[tx] = T.exp2(
                        -T.exp2(A_log[bh] * 1.442695)
                        * softplus_x[tx]
                        * 1.442695
                    )
                if b_is_beta:
                    beta[tx] = b_raw[tx]
                else:
                    beta[tx] = 1.0 / (1.0 + T.exp2(-b_raw[tx] * 1.442695))

            for jv in T.serial(block_DV):
                value_offset = bv * block_DV + jv
                for tx in T.Parallel(64):
                    bn = tx // 32
                    lane = tx % 32
                    h_lane_0[tx] = (
                        h0_source[bn, bh, value_offset, lane] * exp_g[tx]
                    )
                    h_lane_1[tx] = (
                        h0_source[bn, bh, value_offset, lane + 32] * exp_g[tx]
                    )
                    h_lane_2[tx] = (
                        h0_source[bn, bh, value_offset, lane + 64] * exp_g[tx]
                    )
                    h_lane_3[tx] = (
                        h0_source[bn, bh, value_offset, lane + 96] * exp_g[tx]
                    )
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, value_offset] - kv_value[tx]
                    ) * beta[tx]
                    h_lane_0[tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[tx] += k_lane_3[tx] * v_delta[tx]
                    local_sum[tx] = (
                        h_lane_0[tx] * q_lane_0[tx]
                        + h_lane_1[tx] * q_lane_1[tx]
                        + h_lane_2[tx] * q_lane_2[tx]
                        + h_lane_3[tx] * q_lane_3[tx]
                    )
                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if lane == 0:
                        o[0, bn, bh, value_offset] = o_value[tx]
                    h0_source[bn, bh, value_offset, lane] = h_lane_0[tx]
                    h0_source[bn, bh, value_offset, lane + 32] = h_lane_1[tx]
                    h0_source[bn, bh, value_offset, lane + 64] = h_lane_2[tx]
                    h0_source[bn, bh, value_offset, lane + 96] = h_lane_3[tx]

    return tilelang_flashqla_gdn_decode_b2_dual_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_b4_quad_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 16,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, 4, H, DK)
    k_shape = (1, 4, H, DK)
    v_shape = (1, 4, HV, DV)
    a_shape = (1, 4, HV)
    b_shape = (1, 4, HV)
    o_shape = (1, 4, HV, DV)
    state_shape = (4, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_b4_quad_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(HV * num_value_tiles, threads=128) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bhq = bh // (HV // H)

            q_lane_0 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((128,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((128,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((128,), dtype=accum_dtype)
            beta = T.alloc_fragment((128,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((128,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((128,), dtype=accum_dtype)
            x = T.alloc_fragment((128,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((128,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((128,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((128,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((128,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((128,), dtype=accum_dtype)
            o_value = T.alloc_fragment((128,), dtype=accum_dtype)
            bn = T.alloc_var("int32")
            lane = T.alloc_var("int32")
            value_offset = T.alloc_var("int32")

            for tx in T.Parallel(128):
                bn = tx // 32
                lane = tx % 32
                q_lane_0[tx] = q[0, bn, bhq, lane]
                q_lane_1[tx] = q[0, bn, bhq, lane + 32]
                q_lane_2[tx] = q[0, bn, bhq, lane + 64]
                q_lane_3[tx] = q[0, bn, bhq, lane + 96]
                k_lane_0[tx] = k[0, bn, bhq, lane]
                k_lane_1[tx] = k[0, bn, bhq, lane + 32]
                k_lane_2[tx] = k[0, bn, bhq, lane + 64]
                k_lane_3[tx] = k[0, bn, bhq, lane + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]
                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

                a_raw[tx] = a[0, bn, bh]
                b_raw[tx] = b[0, bn, bh]
                if a_is_log_decay:
                    exp_g[tx] = T.exp2(a_raw[tx] * 1.442695)
                else:
                    x[tx] = a_raw[tx] + dt_bias[bh]
                    beta_x[tx] = softplus_beta * x[tx]
                    if beta_x[tx] <= softplus_threshold:
                        softplus_x[tx] = (
                            T.log(1.0 + T.exp2(beta_x[tx] * 1.442695))
                            / softplus_beta
                        )
                    else:
                        softplus_x[tx] = x[tx]
                    exp_g[tx] = T.exp2(
                        -T.exp2(A_log[bh] * 1.442695)
                        * softplus_x[tx]
                        * 1.442695
                    )
                if b_is_beta:
                    beta[tx] = b_raw[tx]
                else:
                    beta[tx] = 1.0 / (1.0 + T.exp2(-b_raw[tx] * 1.442695))

            for jv in T.serial(block_DV):
                value_offset = bv * block_DV + jv
                for tx in T.Parallel(128):
                    bn = tx // 32
                    lane = tx % 32
                    h_lane_0[tx] = (
                        h0_source[bn, bh, value_offset, lane] * exp_g[tx]
                    )
                    h_lane_1[tx] = (
                        h0_source[bn, bh, value_offset, lane + 32] * exp_g[tx]
                    )
                    h_lane_2[tx] = (
                        h0_source[bn, bh, value_offset, lane + 64] * exp_g[tx]
                    )
                    h_lane_3[tx] = (
                        h0_source[bn, bh, value_offset, lane + 96] * exp_g[tx]
                    )
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, value_offset] - kv_value[tx]
                    ) * beta[tx]
                    h_lane_0[tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[tx] += k_lane_3[tx] * v_delta[tx]
                    local_sum[tx] = (
                        h_lane_0[tx] * q_lane_0[tx]
                        + h_lane_1[tx] * q_lane_1[tx]
                        + h_lane_2[tx] * q_lane_2[tx]
                        + h_lane_3[tx] * q_lane_3[tx]
                    )
                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if lane == 0:
                        o[0, bn, bh, value_offset] = o_value[tx]
                    h0_source[bn, bh, value_offset, lane] = h_lane_0[tx]
                    h0_source[bn, bh, value_offset, lane + 32] = h_lane_1[tx]
                    h0_source[bn, bh, value_offset, lane + 64] = h_lane_2[tx]
                    h0_source[bn, bh, value_offset, lane + 96] = h_lane_3[tx]

    return tilelang_flashqla_gdn_decode_b4_quad_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_static_value4_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 16,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
    static_B: int = 4,
):
    warp_tiles = 4
    value_span = block_DV * warp_tiles
    num_value_groups = tilelang.cdiv(DV, value_span)

    q_shape = (1, static_B, H, DK)
    k_shape = (1, static_B, H, DK)
    v_shape = (1, static_B, HV, DV)
    a_shape = (1, static_B, HV)
    b_shape = (1, static_B, HV)
    o_shape = (1, static_B, HV, DV)
    state_shape = (static_B, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_static_value4_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(static_B * HV * num_value_groups, threads=128) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_groups
            bg = bid % num_value_groups
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            q_lane_0 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((128,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((128,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((128,), dtype=accum_dtype)
            beta = T.alloc_fragment((128,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((128,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((128,), dtype=accum_dtype)
            x = T.alloc_fragment((128,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((128,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((128,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((128,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((128,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((128,), dtype=accum_dtype)
            o_value = T.alloc_fragment((128,), dtype=accum_dtype)
            warp = T.alloc_var("int32")
            lane = T.alloc_var("int32")
            value_offset = T.alloc_var("int32")

            for tx in T.Parallel(128):
                warp = tx // 32
                lane = tx % 32
                q_lane_0[tx] = q[0, bn, bhq, lane]
                q_lane_1[tx] = q[0, bn, bhq, lane + 32]
                q_lane_2[tx] = q[0, bn, bhq, lane + 64]
                q_lane_3[tx] = q[0, bn, bhq, lane + 96]
                k_lane_0[tx] = k[0, bn, bhq, lane]
                k_lane_1[tx] = k[0, bn, bhq, lane + 32]
                k_lane_2[tx] = k[0, bn, bhq, lane + 64]
                k_lane_3[tx] = k[0, bn, bhq, lane + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]
                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

                a_raw[tx] = a[0, bn, bh]
                b_raw[tx] = b[0, bn, bh]
                if a_is_log_decay:
                    exp_g[tx] = T.exp2(a_raw[tx] * 1.442695)
                else:
                    x[tx] = a_raw[tx] + dt_bias[bh]
                    beta_x[tx] = softplus_beta * x[tx]
                    if beta_x[tx] <= softplus_threshold:
                        softplus_x[tx] = (
                            T.log(1.0 + T.exp2(beta_x[tx] * 1.442695))
                            / softplus_beta
                        )
                    else:
                        softplus_x[tx] = x[tx]
                    exp_g[tx] = T.exp2(
                        -T.exp2(A_log[bh] * 1.442695)
                        * softplus_x[tx]
                        * 1.442695
                    )
                if b_is_beta:
                    beta[tx] = b_raw[tx]
                else:
                    beta[tx] = 1.0 / (1.0 + T.exp2(-b_raw[tx] * 1.442695))

            for jv in T.serial(block_DV):
                for tx in T.Parallel(128):
                    warp = tx // 32
                    lane = tx % 32
                    value_offset = bg * value_span + warp * block_DV + jv
                    if value_offset < DV:
                        h_lane_0[tx] = (
                            h0_source[bn, bh, value_offset, lane] * exp_g[tx]
                        )
                        h_lane_1[tx] = (
                            h0_source[bn, bh, value_offset, lane + 32]
                            * exp_g[tx]
                        )
                        h_lane_2[tx] = (
                            h0_source[bn, bh, value_offset, lane + 64]
                            * exp_g[tx]
                        )
                        h_lane_3[tx] = (
                            h0_source[bn, bh, value_offset, lane + 96]
                            * exp_g[tx]
                        )
                    else:
                        h_lane_0[tx] = 0.0
                        h_lane_1[tx] = 0.0
                        h_lane_2[tx] = 0.0
                        h_lane_3[tx] = 0.0
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if value_offset < DV:
                        v_delta[tx] = (
                            v[0, bn, bh, value_offset] - kv_value[tx]
                        ) * beta[tx]
                        h_lane_0[tx] += k_lane_0[tx] * v_delta[tx]
                        h_lane_1[tx] += k_lane_1[tx] * v_delta[tx]
                        h_lane_2[tx] += k_lane_2[tx] * v_delta[tx]
                        h_lane_3[tx] += k_lane_3[tx] * v_delta[tx]
                        local_sum[tx] = (
                            h_lane_0[tx] * q_lane_0[tx]
                            + h_lane_1[tx] * q_lane_1[tx]
                            + h_lane_2[tx] * q_lane_2[tx]
                            + h_lane_3[tx] * q_lane_3[tx]
                        )
                        o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                        if lane == 0:
                            o[0, bn, bh, value_offset] = o_value[tx]
                        h0_source[bn, bh, value_offset, lane] = h_lane_0[tx]
                        h0_source[bn, bh, value_offset, lane + 32] = h_lane_1[tx]
                        h0_source[bn, bh, value_offset, lane + 64] = h_lane_2[tx]
                        h0_source[bn, bh, value_offset, lane + 96] = h_lane_3[tx]

    return tilelang_flashqla_gdn_decode_static_value4_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_tile128(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 16,
    identity_indices: bool = False,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_tile128_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=128) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            value_offset = T.alloc_var("int32")
            in_bounds = T.alloc_var("bool")
            if identity_indices:
                state_idx = bn
            else:
                state_idx = h0_indices[bn]

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            q_fragment = T.alloc_fragment((DK,), dtype=accum_dtype)
            k_fragment = T.alloc_fragment((DK,), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_DV,), dtype=accum_dtype)
            kv_fragment = T.alloc_fragment((block_DV,), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_DV,), dtype=accum_dtype)
            reduce_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            q_norm = T.alloc_fragment((1,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((1,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)

            T.clear(h_fragment)
            if state_idx >= 0:
                for jk, jv in T.Parallel(DK, block_DV):
                    value_offset = bv * block_DV + jv
                    in_bounds = value_offset < DV
                    if in_bounds:
                        h_fragment[jk, jv] = h0_source[
                            state_idx, bh, value_offset, jk
                        ]

            for jk in T.Parallel(DK):
                q_fragment[jk] = q[0, bn, bhq, jk]
                k_fragment[jk] = k[0, bn, bhq, jk]

            if use_qk_l2norm_in_kernel:
                q_norm[0] = 0.0
                k_norm[0] = 0.0
                for jk in T.serial(DK):
                    q_norm[0] += q_fragment[jk] * q_fragment[jk]
                    k_norm[0] += k_fragment[jk] * k_fragment[jk]
                q_norm[0] = T.rsqrt(q_norm[0] + 1e-6)
                k_norm[0] = T.rsqrt(k_norm[0] + 1e-6)
                for jk in T.Parallel(DK):
                    q_fragment[jk] *= q_norm[0]
                    k_fragment[jk] *= k_norm[0]
            for jk in T.Parallel(DK):
                q_fragment[jk] *= scale

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            if a_is_log_decay:
                exp_g[0] = T.exp2(a_raw[0] * 1.442695)
            else:
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = (
                        T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                        / softplus_beta
                    )
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp2(
                    -T.exp2(A_log[bh] * 1.442695)
                    * softplus_x[0]
                    * 1.442695
                )
            if b_is_beta:
                beta[0] = b_raw[0]
            else:
                beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for jk, jv in T.Parallel(DK, block_DV):
                h_fragment[jk, jv] *= exp_g[0]
                reduce_fragment[jk, jv] = h_fragment[jk, jv] * k_fragment[jk]
            T.reduce_sum(reduce_fragment, kv_fragment, dim=0, clear=True)

            for jv in T.Parallel(block_DV):
                value_offset = bv * block_DV + jv
                if value_offset < DV:
                    v_fragment[jv] = (
                        v[0, bn, bh, value_offset] - kv_fragment[jv]
                    ) * beta[0]
                else:
                    v_fragment[jv] = 0.0

            for jk, jv in T.Parallel(DK, block_DV):
                h_fragment[jk, jv] += k_fragment[jk] * v_fragment[jv]
                reduce_fragment[jk, jv] = h_fragment[jk, jv] * q_fragment[jk]
            T.reduce_sum(reduce_fragment, o_fragment, dim=0, clear=True)

            if state_idx >= 0:
                for jk, jv in T.Parallel(DK, block_DV):
                    value_offset = bv * block_DV + jv
                    if value_offset < DV:
                        h0_source[state_idx, bh, value_offset, jk] = h_fragment[
                            jk, jv
                        ]

            for jv in T.Parallel(block_DV):
                value_offset = bv * block_DV + jv
                if value_offset < DV:
                    o[0, bn, bh, value_offset] = o_fragment[jv]

    return tilelang_flashqla_gdn_decode_tile128_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_bv16_regular_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 16,
    identity_indices: bool = False,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_bv16_regular_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=32) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            if identity_indices:
                state_idx = bn
            else:
                state_idx = h0_indices[bn]

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            if a_is_log_decay:
                exp_g[0] = T.exp2(a_raw[0] * 1.442695)
            else:
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = (
                        T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                        / softplus_beta
                    )
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp2(
                    -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
                )
            if b_is_beta:
                beta[0] = b_raw[0]
            else:
                beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    if identity_indices:
                        h_lane_0[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx
                        ]
                        h_lane_1[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx + 32
                        ]
                        h_lane_2[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx + 64
                        ]
                        h_lane_3[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx + 96
                        ]
                    elif state_idx >= 0:
                        h_lane_0[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx
                        ]
                        h_lane_1[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 32
                        ]
                        h_lane_2[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 64
                        ]
                        h_lane_3[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 96
                        ]
                    else:
                        h_lane_0[jv, tx] = 0.0
                        h_lane_1[jv, tx] = 0.0
                        h_lane_2[jv, tx] = 0.0
                        h_lane_3[jv, tx] = 0.0

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx]
                q_lane_1[tx] = q[0, bn, bhq, tx + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )

                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]

                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h_lane_0[jv, tx] *= exp_g[0]
                    h_lane_1[jv, tx] *= exp_g[0]
                    h_lane_2[jv, tx] *= exp_g[0]
                    h_lane_3[jv, tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * k_lane_0[tx]
                        + h_lane_1[jv, tx] * k_lane_1[tx]
                        + h_lane_2[jv, tx] * k_lane_2[tx]
                        + h_lane_3[jv, tx] * k_lane_3[tx]
                    )

                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, bv * block_DV + jv] - kv_value[tx]
                    ) * beta[0]

                    h_lane_0[jv, tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[jv, tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[jv, tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[jv, tx] += k_lane_3[tx] * v_delta[tx]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * q_lane_0[tx]
                        + h_lane_1[jv, tx] * q_lane_1[tx]
                        + h_lane_2[jv, tx] * q_lane_2[tx]
                        + h_lane_3[jv, tx] * q_lane_3[tx]
                    )

                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if tx == 0:
                        o[0, bn, bh, bv * block_DV + jv] = o_value[tx]

            if identity_indices:
                for jv in T.serial(block_DV):
                    for tx in T.Parallel(32):
                        h0_source[bn, bh, bv * block_DV + jv, tx] = (
                            h_lane_0[jv, tx]
                        )
                        h0_source[bn, bh, bv * block_DV + jv, tx + 32] = (
                            h_lane_1[jv, tx]
                        )
                        h0_source[bn, bh, bv * block_DV + jv, tx + 64] = (
                            h_lane_2[jv, tx]
                        )
                        h0_source[bn, bh, bv * block_DV + jv, tx + 96] = (
                            h_lane_3[jv, tx]
                        )
            elif state_idx >= 0:
                for jv in T.serial(block_DV):
                    for tx in T.Parallel(32):
                        h0_source[state_idx, bh, bv * block_DV + jv, tx] = (
                            h_lane_0[jv, tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 32] = (
                            h_lane_1[jv, tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 64] = (
                            h_lane_2[jv, tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 96] = (
                            h_lane_3[jv, tx]
                        )

    return tilelang_flashqla_gdn_decode_bv16_regular_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_bv16_split4_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    identity_indices: bool = False,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    block_DV = 16
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_bv16_split4_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=128) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            if identity_indices:
                state_idx = bn
            else:
                state_idx = h0_indices[bn]

            q_lane_0 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((128,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((128,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((128,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((128,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((128,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((128,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((128,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((128,), dtype=accum_dtype)
            o_value = T.alloc_fragment((128,), dtype=accum_dtype)
            lane = T.alloc_var("int32")
            warp = T.alloc_var("int32")
            value_offset = T.alloc_var("int32")

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            if a_is_log_decay:
                exp_g[0] = T.exp2(a_raw[0] * 1.442695)
            else:
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = (
                        T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                        / softplus_beta
                    )
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp2(
                    -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
                )
            if b_is_beta:
                beta[0] = b_raw[0]
            else:
                beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for tx in T.Parallel(128):
                lane = tx % 32
                q_lane_0[tx] = q[0, bn, bhq, lane]
                q_lane_1[tx] = q[0, bn, bhq, lane + 32]
                q_lane_2[tx] = q[0, bn, bhq, lane + 64]
                q_lane_3[tx] = q[0, bn, bhq, lane + 96]
                k_lane_0[tx] = k[0, bn, bhq, lane]
                k_lane_1[tx] = k[0, bn, bhq, lane + 32]
                k_lane_2[tx] = k[0, bn, bhq, lane + 64]
                k_lane_3[tx] = k[0, bn, bhq, lane + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]

                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

                for local_v in T.serial(4):
                    warp = tx // 32
                    value_offset = bv * block_DV + warp * 4 + local_v
                    if identity_indices:
                        h_lane_0[tx] = h0_source[state_idx, bh, value_offset, lane]
                        h_lane_1[tx] = h0_source[state_idx, bh, value_offset, lane + 32]
                        h_lane_2[tx] = h0_source[state_idx, bh, value_offset, lane + 64]
                        h_lane_3[tx] = h0_source[state_idx, bh, value_offset, lane + 96]
                    elif state_idx >= 0:
                        h_lane_0[tx] = h0_source[state_idx, bh, value_offset, lane]
                        h_lane_1[tx] = h0_source[state_idx, bh, value_offset, lane + 32]
                        h_lane_2[tx] = h0_source[state_idx, bh, value_offset, lane + 64]
                        h_lane_3[tx] = h0_source[state_idx, bh, value_offset, lane + 96]
                    else:
                        h_lane_0[tx] = 0.0
                        h_lane_1[tx] = 0.0
                        h_lane_2[tx] = 0.0
                        h_lane_3[tx] = 0.0
                    h_lane_0[tx] *= exp_g[0]
                    h_lane_1[tx] *= exp_g[0]
                    h_lane_2[tx] *= exp_g[0]
                    h_lane_3[tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, value_offset] - kv_value[tx]
                    ) * beta[0]
                    h_lane_0[tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[tx] += k_lane_3[tx] * v_delta[tx]
                    h0_source[state_idx, bh, value_offset, lane] = h_lane_0[tx]
                    h0_source[state_idx, bh, value_offset, lane + 32] = h_lane_1[tx]
                    h0_source[state_idx, bh, value_offset, lane + 64] = h_lane_2[tx]
                    h0_source[state_idx, bh, value_offset, lane + 96] = h_lane_3[tx]
                    local_sum[tx] = (
                        h_lane_0[tx] * q_lane_0[tx]
                        + h_lane_1[tx] * q_lane_1[tx]
                        + h_lane_2[tx] * q_lane_2[tx]
                        + h_lane_3[tx] * q_lane_3[tx]
                    )
                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if lane == 0:
                        o[0, bn, bh, value_offset] = o_value[tx]

    return tilelang_flashqla_gdn_decode_bv16_split4_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_bv16_split2_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    identity_indices: bool = False,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    block_DV = 16
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_bv16_split2_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=64) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            if identity_indices:
                state_idx = bn
            else:
                state_idx = h0_indices[bn]

            q_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((64,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((64,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((64,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((64,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((64,), dtype=accum_dtype)
            o_value = T.alloc_fragment((64,), dtype=accum_dtype)
            lane = T.alloc_var("int32")
            warp = T.alloc_var("int32")
            value_offset = T.alloc_var("int32")

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            if a_is_log_decay:
                exp_g[0] = T.exp2(a_raw[0] * 1.442695)
            else:
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = (
                        T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                        / softplus_beta
                    )
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp2(
                    -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
                )
            if b_is_beta:
                beta[0] = b_raw[0]
            else:
                beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for tx in T.Parallel(64):
                lane = tx % 32
                q_lane_0[tx] = q[0, bn, bhq, lane]
                q_lane_1[tx] = q[0, bn, bhq, lane + 32]
                q_lane_2[tx] = q[0, bn, bhq, lane + 64]
                q_lane_3[tx] = q[0, bn, bhq, lane + 96]
                k_lane_0[tx] = k[0, bn, bhq, lane]
                k_lane_1[tx] = k[0, bn, bhq, lane + 32]
                k_lane_2[tx] = k[0, bn, bhq, lane + 64]
                k_lane_3[tx] = k[0, bn, bhq, lane + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]

                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

                for local_v in T.serial(8):
                    warp = tx // 32
                    value_offset = bv * block_DV + warp * 8 + local_v
                    if identity_indices:
                        h_lane_0[tx] = h0_source[state_idx, bh, value_offset, lane]
                        h_lane_1[tx] = h0_source[state_idx, bh, value_offset, lane + 32]
                        h_lane_2[tx] = h0_source[state_idx, bh, value_offset, lane + 64]
                        h_lane_3[tx] = h0_source[state_idx, bh, value_offset, lane + 96]
                    elif state_idx >= 0:
                        h_lane_0[tx] = h0_source[state_idx, bh, value_offset, lane]
                        h_lane_1[tx] = h0_source[state_idx, bh, value_offset, lane + 32]
                        h_lane_2[tx] = h0_source[state_idx, bh, value_offset, lane + 64]
                        h_lane_3[tx] = h0_source[state_idx, bh, value_offset, lane + 96]
                    else:
                        h_lane_0[tx] = 0.0
                        h_lane_1[tx] = 0.0
                        h_lane_2[tx] = 0.0
                        h_lane_3[tx] = 0.0
                    h_lane_0[tx] *= exp_g[0]
                    h_lane_1[tx] *= exp_g[0]
                    h_lane_2[tx] *= exp_g[0]
                    h_lane_3[tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, value_offset] - kv_value[tx]
                    ) * beta[0]
                    h_lane_0[tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[tx] += k_lane_3[tx] * v_delta[tx]
                    h0_source[state_idx, bh, value_offset, lane] = h_lane_0[tx]
                    h0_source[state_idx, bh, value_offset, lane + 32] = h_lane_1[tx]
                    h0_source[state_idx, bh, value_offset, lane + 64] = h_lane_2[tx]
                    h0_source[state_idx, bh, value_offset, lane + 96] = h_lane_3[tx]
                    local_sum[tx] = (
                        h_lane_0[tx] * q_lane_0[tx]
                        + h_lane_1[tx] * q_lane_1[tx]
                        + h_lane_2[tx] * q_lane_2[tx]
                        + h_lane_3[tx] * q_lane_3[tx]
                    )
                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if lane == 0:
                        o[0, bn, bh, value_offset] = o_value[tx]

    return tilelang_flashqla_gdn_decode_bv16_split2_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_bv16_prepared_gate_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    q_dtype,
    k_dtype,
    v_dtype,
    gate_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 16,
    identity_indices: bool = False,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    gate_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_bv16_prepared_gate_warp_kernel(
        exp_g_pre: T.Tensor(gate_shape, dtype=gate_dtype),
        beta_pre: T.Tensor(gate_shape, dtype=gate_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=32) as (bid,):
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            if identity_indices:
                state_idx = bn
            else:
                state_idx = h0_indices[bn]

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)

            exp_g[0] = exp_g_pre[0, bn, bh]
            beta[0] = beta_pre[0, bn, bh]

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    if identity_indices:
                        h_lane_0[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx
                        ]
                        h_lane_1[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx + 32
                        ]
                        h_lane_2[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx + 64
                        ]
                        h_lane_3[jv, tx] = h0_source[
                            bn, bh, bv * block_DV + jv, tx + 96
                        ]
                    elif state_idx >= 0:
                        h_lane_0[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx
                        ]
                        h_lane_1[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 32
                        ]
                        h_lane_2[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 64
                        ]
                        h_lane_3[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 96
                        ]
                    else:
                        h_lane_0[jv, tx] = 0.0
                        h_lane_1[jv, tx] = 0.0
                        h_lane_2[jv, tx] = 0.0
                        h_lane_3[jv, tx] = 0.0

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx]
                q_lane_1[tx] = q[0, bn, bhq, tx + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )

                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]

                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h_lane_0[jv, tx] *= exp_g[0]
                    h_lane_1[jv, tx] *= exp_g[0]
                    h_lane_2[jv, tx] *= exp_g[0]
                    h_lane_3[jv, tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * k_lane_0[tx]
                        + h_lane_1[jv, tx] * k_lane_1[tx]
                        + h_lane_2[jv, tx] * k_lane_2[tx]
                        + h_lane_3[jv, tx] * k_lane_3[tx]
                    )

                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (
                        v[0, bn, bh, bv * block_DV + jv] - kv_value[tx]
                    ) * beta[0]

                    h_lane_0[jv, tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[jv, tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[jv, tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[jv, tx] += k_lane_3[tx] * v_delta[tx]
                    local_sum[tx] = (
                        h_lane_0[jv, tx] * q_lane_0[tx]
                        + h_lane_1[jv, tx] * q_lane_1[tx]
                        + h_lane_2[jv, tx] * q_lane_2[tx]
                        + h_lane_3[jv, tx] * q_lane_3[tx]
                    )

                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if tx == 0:
                        o[0, bn, bh, bv * block_DV + jv] = o_value[tx]

            if identity_indices:
                for jv in T.serial(block_DV):
                    for tx in T.Parallel(32):
                        h0_source[bn, bh, bv * block_DV + jv, tx] = (
                            h_lane_0[jv, tx]
                        )
                        h0_source[bn, bh, bv * block_DV + jv, tx + 32] = (
                            h_lane_1[jv, tx]
                        )
                        h0_source[bn, bh, bv * block_DV + jv, tx + 64] = (
                            h_lane_2[jv, tx]
                        )
                        h0_source[bn, bh, bv * block_DV + jv, tx + 96] = (
                            h_lane_3[jv, tx]
                        )
            elif state_idx >= 0:
                for jv in T.serial(block_DV):
                    for tx in T.Parallel(32):
                        h0_source[state_idx, bh, bv * block_DV + jv, tx] = (
                            h_lane_0[jv, tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 32] = (
                            h_lane_1[jv, tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 64] = (
                            h_lane_2[jv, tx]
                        )
                        h0_source[state_idx, bh, bv * block_DV + jv, tx + 96] = (
                            h_lane_3[jv, tx]
                        )

    return tilelang_flashqla_gdn_decode_bv16_prepared_gate_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_gqa3_bv16_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    identity_indices: bool = False,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    block_DV = 16
    num_value_tiles = tilelang.cdiv(DV, block_DV)
    group_size = HV // H

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_gqa3_bv16_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * H * num_value_tiles, threads=96) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnk = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnk // H
            bhq = bnk % H

            state_idx = T.alloc_var("int32")
            if identity_indices:
                state_idx = bn
            else:
                state_idx = h0_indices[bn]

            q_lane_0 = T.alloc_fragment((96,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((96,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((96,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((96,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((96,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((96,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((96,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((96,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((96,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((96,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((96,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((96,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((96,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((96,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((96,), dtype=accum_dtype)
            beta = T.alloc_fragment((96,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((96,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((96,), dtype=accum_dtype)
            x = T.alloc_fragment((96,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((96,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((96,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((96,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((96,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((96,), dtype=accum_dtype)
            o_value = T.alloc_fragment((96,), dtype=accum_dtype)

            for t in T.Parallel(96):
                i_g = t // 32
                t_lane = t % 32
                t_bh = bhq * group_size + i_g
                q_lane_0[t] = q[0, bn, bhq, t_lane]
                q_lane_1[t] = q[0, bn, bhq, t_lane + 32]
                q_lane_2[t] = q[0, bn, bhq, t_lane + 64]
                q_lane_3[t] = q[0, bn, bhq, t_lane + 96]
                k_lane_0[t] = k[0, bn, bhq, t_lane]
                k_lane_1[t] = k[0, bn, bhq, t_lane + 32]
                k_lane_2[t] = k[0, bn, bhq, t_lane + 64]
                k_lane_3[t] = k[0, bn, bhq, t_lane + 96]
                q_norm[t] = (
                    q_lane_0[t] * q_lane_0[t]
                    + q_lane_1[t] * q_lane_1[t]
                    + q_lane_2[t] * q_lane_2[t]
                    + q_lane_3[t] * q_lane_3[t]
                )
                k_norm[t] = (
                    k_lane_0[t] * k_lane_0[t]
                    + k_lane_1[t] * k_lane_1[t]
                    + k_lane_2[t] * k_lane_2[t]
                    + k_lane_3[t] * k_lane_3[t]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[t] = T.warp_reduce_sum(q_norm[t])
                    k_norm[t] = T.warp_reduce_sum(k_norm[t])
                    q_norm[t] = T.rsqrt(q_norm[t] + 1e-6)
                    k_norm[t] = T.rsqrt(k_norm[t] + 1e-6)
                    q_lane_0[t] *= q_norm[t]
                    q_lane_1[t] *= q_norm[t]
                    q_lane_2[t] *= q_norm[t]
                    q_lane_3[t] *= q_norm[t]
                    k_lane_0[t] *= k_norm[t]
                    k_lane_1[t] *= k_norm[t]
                    k_lane_2[t] *= k_norm[t]
                    k_lane_3[t] *= k_norm[t]
                q_lane_0[t] *= scale
                q_lane_1[t] *= scale
                q_lane_2[t] *= scale
                q_lane_3[t] *= scale

                a_raw[t] = a[0, bn, t_bh]
                b_raw[t] = b[0, bn, t_bh]
                if a_is_log_decay:
                    exp_g[t] = T.exp2(a_raw[t] * 1.442695)
                else:
                    x[t] = a_raw[t] + dt_bias[t_bh]
                    beta_x[t] = softplus_beta * x[t]
                    if beta_x[t] <= softplus_threshold:
                        softplus_x[t] = (
                            T.log(1.0 + T.exp2(beta_x[t] * 1.442695))
                            / softplus_beta
                        )
                    else:
                        softplus_x[t] = x[t]
                    exp_g[t] = T.exp2(
                        -T.exp2(A_log[t_bh] * 1.442695)
                        * softplus_x[t]
                        * 1.442695
                    )
                if b_is_beta:
                    beta[t] = b_raw[t]
                else:
                    beta[t] = 1.0 / (1.0 + T.exp2(-b_raw[t] * 1.442695))

            for jv in T.serial(block_DV):
                value_offset = bv * block_DV + jv
                for t in T.Parallel(96):
                    i_g = t // 32
                    t_lane = t % 32
                    t_bh = bhq * group_size + i_g
                    if identity_indices:
                        h_lane_0[t] = h0_source[bn, t_bh, value_offset, t_lane]
                        h_lane_1[t] = h0_source[
                            bn, t_bh, value_offset, t_lane + 32
                        ]
                        h_lane_2[t] = h0_source[
                            bn, t_bh, value_offset, t_lane + 64
                        ]
                        h_lane_3[t] = h0_source[
                            bn, t_bh, value_offset, t_lane + 96
                        ]
                    elif state_idx >= 0:
                        h_lane_0[t] = h0_source[
                            state_idx, t_bh, value_offset, t_lane
                        ]
                        h_lane_1[t] = h0_source[
                            state_idx, t_bh, value_offset, t_lane + 32
                        ]
                        h_lane_2[t] = h0_source[
                            state_idx, t_bh, value_offset, t_lane + 64
                        ]
                        h_lane_3[t] = h0_source[
                            state_idx, t_bh, value_offset, t_lane + 96
                        ]
                    else:
                        h_lane_0[t] = 0.0
                        h_lane_1[t] = 0.0
                        h_lane_2[t] = 0.0
                        h_lane_3[t] = 0.0

                    h_lane_0[t] *= exp_g[t]
                    h_lane_1[t] *= exp_g[t]
                    h_lane_2[t] *= exp_g[t]
                    h_lane_3[t] *= exp_g[t]
                    local_sum[t] = (
                        h_lane_0[t] * k_lane_0[t]
                        + h_lane_1[t] * k_lane_1[t]
                        + h_lane_2[t] * k_lane_2[t]
                        + h_lane_3[t] * k_lane_3[t]
                    )
                    kv_value[t] = T.warp_reduce_sum(local_sum[t])
                    v_delta[t] = (
                        v[0, bn, t_bh, value_offset] - kv_value[t]
                    ) * beta[t]
                    h_lane_0[t] += k_lane_0[t] * v_delta[t]
                    h_lane_1[t] += k_lane_1[t] * v_delta[t]
                    h_lane_2[t] += k_lane_2[t] * v_delta[t]
                    h_lane_3[t] += k_lane_3[t] * v_delta[t]
                    if identity_indices:
                        h0_source[bn, t_bh, value_offset, t_lane] = h_lane_0[t]
                        h0_source[bn, t_bh, value_offset, t_lane + 32] = h_lane_1[t]
                        h0_source[bn, t_bh, value_offset, t_lane + 64] = h_lane_2[t]
                        h0_source[bn, t_bh, value_offset, t_lane + 96] = h_lane_3[t]
                    elif state_idx >= 0:
                        h0_source[state_idx, t_bh, value_offset, t_lane] = h_lane_0[t]
                        h0_source[
                            state_idx, t_bh, value_offset, t_lane + 32
                        ] = h_lane_1[t]
                        h0_source[
                            state_idx, t_bh, value_offset, t_lane + 64
                        ] = h_lane_2[t]
                        h0_source[
                            state_idx, t_bh, value_offset, t_lane + 96
                        ] = h_lane_3[t]
                    local_sum[t] = (
                        h_lane_0[t] * q_lane_0[t]
                        + h_lane_1[t] * q_lane_1[t]
                        + h_lane_2[t] * q_lane_2[t]
                        + h_lane_3[t] * q_lane_3[t]
                    )
                    o_value[t] = T.warp_reduce_sum(local_sum[t])
                    if t_lane == 0:
                        o[0, bn, t_bh, value_offset] = o_value[t]

    return tilelang_flashqla_gdn_decode_gqa3_bv16_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_gqa_serial_bv16_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 16,
    identity_indices: bool = False,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    num_value_tiles = tilelang.cdiv(DV, block_DV)
    group_size = HV // H

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_gqa_serial_bv16_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * H * num_value_tiles, threads=32) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnk = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnk // H
            bhq = bnk % H

            state_idx = T.alloc_var("int32")
            if identity_indices:
                state_idx = bn
            else:
                state_idx = h0_indices[bn]

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx]
                q_lane_1[tx] = q[0, bn, bhq, tx + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]
                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

            for ig in T.serial(group_size):
                bh = bhq * group_size + ig
                a_raw[0] = a[0, bn, bh]
                b_raw[0] = b[0, bn, bh]
                if a_is_log_decay:
                    exp_g[0] = T.exp2(a_raw[0] * 1.442695)
                else:
                    x[0] = a_raw[0] + dt_bias[bh]
                    beta_x[0] = softplus_beta * x[0]
                    if beta_x[0] <= softplus_threshold:
                        softplus_x[0] = (
                            T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                            / softplus_beta
                        )
                    else:
                        softplus_x[0] = x[0]
                    exp_g[0] = T.exp2(
                        -T.exp2(A_log[bh] * 1.442695)
                        * softplus_x[0]
                        * 1.442695
                    )
                if b_is_beta:
                    beta[0] = b_raw[0]
                else:
                    beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

                for jv in T.serial(block_DV):
                    for tx in T.Parallel(32):
                        if identity_indices:
                            h_lane_0[jv, tx] = h0_source[
                                bn, bh, bv * block_DV + jv, tx
                            ]
                            h_lane_1[jv, tx] = h0_source[
                                bn, bh, bv * block_DV + jv, tx + 32
                            ]
                            h_lane_2[jv, tx] = h0_source[
                                bn, bh, bv * block_DV + jv, tx + 64
                            ]
                            h_lane_3[jv, tx] = h0_source[
                                bn, bh, bv * block_DV + jv, tx + 96
                            ]
                        elif state_idx >= 0:
                            h_lane_0[jv, tx] = h0_source[
                                state_idx, bh, bv * block_DV + jv, tx
                            ]
                            h_lane_1[jv, tx] = h0_source[
                                state_idx, bh, bv * block_DV + jv, tx + 32
                            ]
                            h_lane_2[jv, tx] = h0_source[
                                state_idx, bh, bv * block_DV + jv, tx + 64
                            ]
                            h_lane_3[jv, tx] = h0_source[
                                state_idx, bh, bv * block_DV + jv, tx + 96
                            ]
                        else:
                            h_lane_0[jv, tx] = 0.0
                            h_lane_1[jv, tx] = 0.0
                            h_lane_2[jv, tx] = 0.0
                            h_lane_3[jv, tx] = 0.0

                for jv in T.serial(block_DV):
                    for tx in T.Parallel(32):
                        h_lane_0[jv, tx] *= exp_g[0]
                        h_lane_1[jv, tx] *= exp_g[0]
                        h_lane_2[jv, tx] *= exp_g[0]
                        h_lane_3[jv, tx] *= exp_g[0]
                        local_sum[tx] = (
                            h_lane_0[jv, tx] * k_lane_0[tx]
                            + h_lane_1[jv, tx] * k_lane_1[tx]
                            + h_lane_2[jv, tx] * k_lane_2[tx]
                            + h_lane_3[jv, tx] * k_lane_3[tx]
                        )

                        kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                        v_delta[tx] = (
                            v[0, bn, bh, bv * block_DV + jv] - kv_value[tx]
                        ) * beta[0]

                        h_lane_0[jv, tx] += k_lane_0[tx] * v_delta[tx]
                        h_lane_1[jv, tx] += k_lane_1[tx] * v_delta[tx]
                        h_lane_2[jv, tx] += k_lane_2[tx] * v_delta[tx]
                        h_lane_3[jv, tx] += k_lane_3[tx] * v_delta[tx]
                        local_sum[tx] = (
                            h_lane_0[jv, tx] * q_lane_0[tx]
                            + h_lane_1[jv, tx] * q_lane_1[tx]
                            + h_lane_2[jv, tx] * q_lane_2[tx]
                            + h_lane_3[jv, tx] * q_lane_3[tx]
                        )

                        o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                        if tx == 0:
                            o[0, bn, bh, bv * block_DV + jv] = o_value[tx]

                if identity_indices:
                    for jv in T.serial(block_DV):
                        for tx in T.Parallel(32):
                            h0_source[bn, bh, bv * block_DV + jv, tx] = (
                                h_lane_0[jv, tx]
                            )
                            h0_source[bn, bh, bv * block_DV + jv, tx + 32] = (
                                h_lane_1[jv, tx]
                            )
                            h0_source[bn, bh, bv * block_DV + jv, tx + 64] = (
                                h_lane_2[jv, tx]
                            )
                            h0_source[bn, bh, bv * block_DV + jv, tx + 96] = (
                                h_lane_3[jv, tx]
                            )
                elif state_idx >= 0:
                    for jv in T.serial(block_DV):
                        for tx in T.Parallel(32):
                            h0_source[state_idx, bh, bv * block_DV + jv, tx] = (
                                h_lane_0[jv, tx]
                            )
                            h0_source[
                                state_idx, bh, bv * block_DV + jv, tx + 32
                            ] = h_lane_1[jv, tx]
                            h0_source[
                                state_idx, bh, bv * block_DV + jv, tx + 64
                            ] = h_lane_2[jv, tx]
                            h0_source[
                                state_idx, bh, bv * block_DV + jv, tx + 96
                            ] = h_lane_3[jv, tx]

    return tilelang_flashqla_gdn_decode_gqa_serial_bv16_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def tilelang_flashqla_gdn_decode_bv16_stream_warp(
    H,
    HV,
    DK,
    DV,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 16,
    identity_indices: bool = False,
    max_nreg: int = 0,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_decode_bv16_stream_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=32) as (bid,):
            if max_nreg > 0:
                T.set_max_nreg(max_nreg, 1)
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            state_idx = T.alloc_var("int32")
            if identity_indices:
                state_idx = bn
            else:
                state_idx = h0_indices[bn]

            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)

            a_raw[0] = a[0, bn, bh]
            b_raw[0] = b[0, bn, bh]
            if a_is_log_decay:
                exp_g[0] = T.exp2(a_raw[0] * 1.442695)
            else:
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = (
                        T.log(1.0 + T.exp2(beta_x[0] * 1.442695))
                        / softplus_beta
                    )
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp2(
                    -T.exp2(A_log[bh] * 1.442695) * softplus_x[0] * 1.442695
                )
            if b_is_beta:
                beta[0] = b_raw[0]
            else:
                beta[0] = 1.0 / (1.0 + T.exp2(-b_raw[0] * 1.442695))

            for tx in T.Parallel(32):
                q_lane_0[tx] = q[0, bn, bhq, tx]
                q_lane_1[tx] = q[0, bn, bhq, tx + 32]
                q_lane_2[tx] = q[0, bn, bhq, tx + 64]
                q_lane_3[tx] = q[0, bn, bhq, tx + 96]
                k_lane_0[tx] = k[0, bn, bhq, tx]
                k_lane_1[tx] = k[0, bn, bhq, tx + 32]
                k_lane_2[tx] = k[0, bn, bhq, tx + 64]
                k_lane_3[tx] = k[0, bn, bhq, tx + 96]
                q_norm[tx] = (
                    q_lane_0[tx] * q_lane_0[tx]
                    + q_lane_1[tx] * q_lane_1[tx]
                    + q_lane_2[tx] * q_lane_2[tx]
                    + q_lane_3[tx] * q_lane_3[tx]
                )
                k_norm[tx] = (
                    k_lane_0[tx] * k_lane_0[tx]
                    + k_lane_1[tx] * k_lane_1[tx]
                    + k_lane_2[tx] * k_lane_2[tx]
                    + k_lane_3[tx] * k_lane_3[tx]
                )
                if use_qk_l2norm_in_kernel:
                    q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                    k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                    q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                    k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                    q_lane_0[tx] *= q_norm[tx]
                    q_lane_1[tx] *= q_norm[tx]
                    q_lane_2[tx] *= q_norm[tx]
                    q_lane_3[tx] *= q_norm[tx]
                    k_lane_0[tx] *= k_norm[tx]
                    k_lane_1[tx] *= k_norm[tx]
                    k_lane_2[tx] *= k_norm[tx]
                    k_lane_3[tx] *= k_norm[tx]
                q_lane_0[tx] *= scale
                q_lane_1[tx] *= scale
                q_lane_2[tx] *= scale
                q_lane_3[tx] *= scale

            for jv in T.serial(block_DV):
                value_offset = bv * block_DV + jv
                for tx in T.Parallel(32):
                    if identity_indices:
                        h_lane_0[tx] = h0_source[bn, bh, value_offset, tx]
                        h_lane_1[tx] = h0_source[bn, bh, value_offset, tx + 32]
                        h_lane_2[tx] = h0_source[bn, bh, value_offset, tx + 64]
                        h_lane_3[tx] = h0_source[bn, bh, value_offset, tx + 96]
                    elif state_idx >= 0:
                        h_lane_0[tx] = h0_source[state_idx, bh, value_offset, tx]
                        h_lane_1[tx] = h0_source[
                            state_idx, bh, value_offset, tx + 32
                        ]
                        h_lane_2[tx] = h0_source[
                            state_idx, bh, value_offset, tx + 64
                        ]
                        h_lane_3[tx] = h0_source[
                            state_idx, bh, value_offset, tx + 96
                        ]
                    else:
                        h_lane_0[tx] = 0.0
                        h_lane_1[tx] = 0.0
                        h_lane_2[tx] = 0.0
                        h_lane_3[tx] = 0.0
                    h_lane_0[tx] *= exp_g[0]
                    h_lane_1[tx] *= exp_g[0]
                    h_lane_2[tx] *= exp_g[0]
                    h_lane_3[tx] *= exp_g[0]
                    local_sum[tx] = (
                        h_lane_0[tx] * k_lane_0[tx]
                        + h_lane_1[tx] * k_lane_1[tx]
                        + h_lane_2[tx] * k_lane_2[tx]
                        + h_lane_3[tx] * k_lane_3[tx]
                    )
                    kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    v_delta[tx] = (v[0, bn, bh, value_offset] - kv_value[tx]) * beta[0]
                    h_lane_0[tx] += k_lane_0[tx] * v_delta[tx]
                    h_lane_1[tx] += k_lane_1[tx] * v_delta[tx]
                    h_lane_2[tx] += k_lane_2[tx] * v_delta[tx]
                    h_lane_3[tx] += k_lane_3[tx] * v_delta[tx]
                    if identity_indices:
                        h0_source[bn, bh, value_offset, tx] = h_lane_0[tx]
                        h0_source[bn, bh, value_offset, tx + 32] = h_lane_1[tx]
                        h0_source[bn, bh, value_offset, tx + 64] = h_lane_2[tx]
                        h0_source[bn, bh, value_offset, tx + 96] = h_lane_3[tx]
                    elif state_idx >= 0:
                        h0_source[state_idx, bh, value_offset, tx] = h_lane_0[tx]
                        h0_source[state_idx, bh, value_offset, tx + 32] = h_lane_1[tx]
                        h0_source[state_idx, bh, value_offset, tx + 64] = h_lane_2[tx]
                        h0_source[state_idx, bh, value_offset, tx + 96] = h_lane_3[tx]
                    local_sum[tx] = (
                        h_lane_0[tx] * q_lane_0[tx]
                        + h_lane_1[tx] * q_lane_1[tx]
                        + h_lane_2[tx] * q_lane_2[tx]
                        + h_lane_3[tx] * q_lane_3[tx]
                    )
                    o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                    if tx == 0:
                        o[0, bn, bh, value_offset] = o_value[tx]

    return tilelang_flashqla_gdn_decode_bv16_stream_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_flashqla_gdn_regular_bv32_warp(
    H,
    HV,
    DK,
    DV,
    tokens_per_seq,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    disable_state_update,
    cache_intermediate_states,
    has_tree_attention,
    block_DV: int = 32,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    cache_steps = T.dynamic("cache_steps")
    num_value_tiles = tilelang.cdiv(DV, block_DV)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)
    intermediate_shape = (num_sequences, cache_steps, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_regular_bv32_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        intermediate_states_buffer: T.Tensor(intermediate_shape, dtype=state_dtype),
        intermediate_state_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        retrieve_parent_token: T.Tensor((num_sequences, cache_steps), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tiles, threads=32) as (bid,):
            bnh = bid // num_value_tiles
            bv = bid % num_value_tiles
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            seq_start = T.alloc_var("int32")
            state_idx = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            token_idx = T.alloc_var("int32")

            seq_start = bn * tokens_per_seq
            state_idx = h0_indices[bn]
            cache_idx = -1
            if cache_intermediate_states:
                cache_idx = intermediate_state_indices[bn]

            # DK is fixed to 128 by the Python wrapper, so each warp lane owns four K values.
            q_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((32,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((32,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((block_DV, 32), dtype=accum_dtype)
            q_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((32,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((1,), dtype=accum_dtype)
            beta = T.alloc_fragment((1,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((1,), dtype=accum_dtype)
            x = T.alloc_fragment((1,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((1,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((1,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((32,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((32,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((32,), dtype=accum_dtype)
            o_value = T.alloc_fragment((32,), dtype=accum_dtype)
            parent_step = T.alloc_fragment((1,), dtype=indices_dtype)

            for jv in T.serial(block_DV):
                for tx in T.Parallel(32):
                    h_lane_0[jv, tx] = 0.0
                    h_lane_1[jv, tx] = 0.0
                    h_lane_2[jv, tx] = 0.0
                    h_lane_3[jv, tx] = 0.0
                    if state_idx >= 0:
                        h_lane_0[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx
                        ]
                        h_lane_1[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 32
                        ]
                        h_lane_2[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 64
                        ]
                        h_lane_3[jv, tx] = h0_source[
                            state_idx, bh, bv * block_DV + jv, tx + 96
                        ]

            for step in T.serial(tokens_per_seq):
                token_idx = seq_start + step

                a_raw[0] = a[0, token_idx, bh]
                b_raw[0] = b[0, token_idx, bh]
                x[0] = a_raw[0] + dt_bias[bh]
                beta_x[0] = softplus_beta * x[0]
                if beta_x[0] <= softplus_threshold:
                    softplus_x[0] = T.log(1.0 + T.exp(beta_x[0])) / softplus_beta
                else:
                    softplus_x[0] = x[0]
                exp_g[0] = T.exp(-T.exp(A_log[bh]) * softplus_x[0])
                beta[0] = 1.0 / (1.0 + T.exp(-b_raw[0]))

                if has_tree_attention:
                    if step != 0 and cache_idx >= 0:
                        parent_step[0] = retrieve_parent_token[bn, step]
                        if parent_step[0] != step - 1:
                            for jv in T.serial(block_DV):
                                for tx in T.Parallel(32):
                                    h_lane_0[jv, tx] = intermediate_states_buffer[
                                        cache_idx,
                                        parent_step[0],
                                        bh,
                                        bv * block_DV + jv,
                                        tx,
                                    ]
                                    h_lane_1[jv, tx] = intermediate_states_buffer[
                                        cache_idx,
                                        parent_step[0],
                                        bh,
                                        bv * block_DV + jv,
                                        tx + 32,
                                    ]
                                    h_lane_2[jv, tx] = intermediate_states_buffer[
                                        cache_idx,
                                        parent_step[0],
                                        bh,
                                        bv * block_DV + jv,
                                        tx + 64,
                                    ]
                                    h_lane_3[jv, tx] = intermediate_states_buffer[
                                        cache_idx,
                                        parent_step[0],
                                        bh,
                                        bv * block_DV + jv,
                                        tx + 96,
                                    ]

                for tx in T.Parallel(32):
                    q_lane_0[tx] = q[0, token_idx, bhq, tx]
                    q_lane_1[tx] = q[0, token_idx, bhq, tx + 32]
                    q_lane_2[tx] = q[0, token_idx, bhq, tx + 64]
                    q_lane_3[tx] = q[0, token_idx, bhq, tx + 96]
                    k_lane_0[tx] = k[0, token_idx, bhq, tx]
                    k_lane_1[tx] = k[0, token_idx, bhq, tx + 32]
                    k_lane_2[tx] = k[0, token_idx, bhq, tx + 64]
                    k_lane_3[tx] = k[0, token_idx, bhq, tx + 96]
                    q_norm[tx] = (
                        q_lane_0[tx] * q_lane_0[tx]
                        + q_lane_1[tx] * q_lane_1[tx]
                        + q_lane_2[tx] * q_lane_2[tx]
                        + q_lane_3[tx] * q_lane_3[tx]
                    )
                    k_norm[tx] = (
                        k_lane_0[tx] * k_lane_0[tx]
                        + k_lane_1[tx] * k_lane_1[tx]
                        + k_lane_2[tx] * k_lane_2[tx]
                        + k_lane_3[tx] * k_lane_3[tx]
                    )

                    if use_qk_l2norm_in_kernel:
                        q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                        k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                        q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                        k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                        q_lane_0[tx] *= q_norm[tx]
                        q_lane_1[tx] *= q_norm[tx]
                        q_lane_2[tx] *= q_norm[tx]
                        q_lane_3[tx] *= q_norm[tx]
                        k_lane_0[tx] *= k_norm[tx]
                        k_lane_1[tx] *= k_norm[tx]
                        k_lane_2[tx] *= k_norm[tx]
                        k_lane_3[tx] *= k_norm[tx]

                    q_lane_0[tx] *= scale
                    q_lane_1[tx] *= scale
                    q_lane_2[tx] *= scale
                    q_lane_3[tx] *= scale

                for jv in T.serial(block_DV):
                    for tx in T.Parallel(32):
                        h_lane_0[jv, tx] *= exp_g[0]
                        h_lane_1[jv, tx] *= exp_g[0]
                        h_lane_2[jv, tx] *= exp_g[0]
                        h_lane_3[jv, tx] *= exp_g[0]
                        local_sum[tx] = (
                            h_lane_0[jv, tx] * k_lane_0[tx]
                            + h_lane_1[jv, tx] * k_lane_1[tx]
                            + h_lane_2[jv, tx] * k_lane_2[tx]
                            + h_lane_3[jv, tx] * k_lane_3[tx]
                        )

                        kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                        v_delta[tx] = (
                            v[0, token_idx, bh, bv * block_DV + jv] - kv_value[tx]
                        ) * beta[0]

                        h_lane_0[jv, tx] += k_lane_0[tx] * v_delta[tx]
                        h_lane_1[jv, tx] += k_lane_1[tx] * v_delta[tx]
                        h_lane_2[jv, tx] += k_lane_2[tx] * v_delta[tx]
                        h_lane_3[jv, tx] += k_lane_3[tx] * v_delta[tx]
                        local_sum[tx] = (
                            h_lane_0[jv, tx] * q_lane_0[tx]
                            + h_lane_1[jv, tx] * q_lane_1[tx]
                            + h_lane_2[jv, tx] * q_lane_2[tx]
                            + h_lane_3[jv, tx] * q_lane_3[tx]
                        )

                        o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                        if tx == 0:
                            o[0, token_idx, bh, bv * block_DV + jv] = o_value[tx]

                        if cache_intermediate_states:
                            if cache_idx >= 0:
                                intermediate_states_buffer[
                                    cache_idx, step, bh, bv * block_DV + jv, tx
                                ] = h_lane_0[jv, tx]
                                intermediate_states_buffer[
                                    cache_idx, step, bh, bv * block_DV + jv, tx + 32
                                ] = h_lane_1[jv, tx]
                                intermediate_states_buffer[
                                    cache_idx, step, bh, bv * block_DV + jv, tx + 64
                                ] = h_lane_2[jv, tx]
                                intermediate_states_buffer[
                                    cache_idx, step, bh, bv * block_DV + jv, tx + 96
                                ] = h_lane_3[jv, tx]

            if not disable_state_update:
                if state_idx >= 0:
                    for jv in T.serial(block_DV):
                        for tx in T.Parallel(32):
                            h0_source[state_idx, bh, bv * block_DV + jv, tx] = (
                                h_lane_0[jv, tx]
                            )
                            h0_source[
                                state_idx, bh, bv * block_DV + jv, tx + 32
                            ] = h_lane_1[jv, tx]
                            h0_source[
                                state_idx, bh, bv * block_DV + jv, tx + 64
                            ] = h_lane_2[jv, tx]
                            h0_source[
                                state_idx, bh, bv * block_DV + jv, tx + 96
                            ] = h_lane_3[jv, tx]

    return tilelang_flashqla_gdn_regular_bv32_warp_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_flashqla_gdn_tree_verify_bv8x2_warp(
    H,
    HV,
    DK,
    DV,
    tokens_per_seq,
    scale,
    softplus_beta,
    softplus_threshold,
    q_dtype,
    k_dtype,
    v_dtype,
    a_dtype,
    b_dtype,
    a_log_dtype,
    dt_bias_dtype,
    state_dtype,
    indices_dtype,
    o_dtype,
    accum_dtype,
    use_qk_l2norm_in_kernel,
    block_DV: int = 8,
):
    total_tokens = T.dynamic("total_tokens")
    num_sequences = T.dynamic("num_sequences")
    cache_steps = T.dynamic("cache_steps")
    value_span = block_DV * 2
    num_value_tile_groups = tilelang.cdiv(DV, value_span)

    q_shape = (1, total_tokens, H, DK)
    k_shape = (1, total_tokens, H, DK)
    v_shape = (1, total_tokens, HV, DV)
    a_shape = (1, total_tokens, HV)
    b_shape = (1, total_tokens, HV)
    o_shape = (1, total_tokens, HV, DV)
    state_shape = (num_sequences, HV, DV, DK)
    intermediate_shape = (num_sequences, cache_steps, HV, DV, DK)

    @T.prim_func
    def tilelang_flashqla_gdn_tree_verify_bv8x2_warp_kernel(
        A_log: T.Tensor((HV,), dtype=a_log_dtype),
        a: T.Tensor(a_shape, dtype=a_dtype),
        dt_bias: T.Tensor((HV,), dtype=dt_bias_dtype),
        q: T.Tensor(q_shape, dtype=q_dtype),
        k: T.Tensor(k_shape, dtype=k_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0_source: T.Tensor(state_shape, dtype=state_dtype),
        h0_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        intermediate_states_buffer: T.Tensor(intermediate_shape, dtype=state_dtype),
        intermediate_state_indices: T.Tensor((num_sequences,), dtype=indices_dtype),
        retrieve_parent_token: T.Tensor((num_sequences, cache_steps), dtype=indices_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(num_sequences * HV * num_value_tile_groups, threads=64) as (bid,):
            bnh = bid // num_value_tile_groups
            bv = bid % num_value_tile_groups
            bn = bnh // HV
            bh = bnh % HV
            bhq = bh // (HV // H)

            seq_start = T.alloc_var("int32")
            state_idx = T.alloc_var("int32")
            cache_idx = T.alloc_var("int32")
            token_idx = T.alloc_var("int32")

            seq_start = bn * tokens_per_seq
            state_idx = h0_indices[bn]
            cache_idx = intermediate_state_indices[bn]

            q_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            q_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_0 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_1 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_2 = T.alloc_fragment((64,), dtype=accum_dtype)
            k_lane_3 = T.alloc_fragment((64,), dtype=accum_dtype)
            h_lane_0 = T.alloc_fragment((block_DV, 64), dtype=accum_dtype)
            h_lane_1 = T.alloc_fragment((block_DV, 64), dtype=accum_dtype)
            h_lane_2 = T.alloc_fragment((block_DV, 64), dtype=accum_dtype)
            h_lane_3 = T.alloc_fragment((block_DV, 64), dtype=accum_dtype)
            q_norm = T.alloc_fragment((64,), dtype=accum_dtype)
            k_norm = T.alloc_fragment((64,), dtype=accum_dtype)
            exp_g = T.alloc_fragment((64,), dtype=accum_dtype)
            beta = T.alloc_fragment((64,), dtype=accum_dtype)
            a_raw = T.alloc_fragment((64,), dtype=accum_dtype)
            b_raw = T.alloc_fragment((64,), dtype=accum_dtype)
            x = T.alloc_fragment((64,), dtype=accum_dtype)
            beta_x = T.alloc_fragment((64,), dtype=accum_dtype)
            softplus_x = T.alloc_fragment((64,), dtype=accum_dtype)
            local_sum = T.alloc_fragment((64,), dtype=accum_dtype)
            kv_value = T.alloc_fragment((64,), dtype=accum_dtype)
            v_delta = T.alloc_fragment((64,), dtype=accum_dtype)
            o_value = T.alloc_fragment((64,), dtype=accum_dtype)
            parent_step = T.alloc_fragment((1,), dtype=indices_dtype)

            for jv in T.serial(block_DV):
                for tx in T.Parallel(64):
                    h_lane_0[jv, tx] = 0.0
                    h_lane_1[jv, tx] = 0.0
                    h_lane_2[jv, tx] = 0.0
                    h_lane_3[jv, tx] = 0.0
                    if state_idx >= 0:
                        if bv * value_span + (tx // 32) * block_DV + jv < DV:
                            h_lane_0[jv, tx] = h0_source[
                                state_idx,
                                bh,
                                bv * value_span + (tx // 32) * block_DV + jv,
                                tx % 32,
                            ]
                            h_lane_1[jv, tx] = h0_source[
                                state_idx,
                                bh,
                                bv * value_span + (tx // 32) * block_DV + jv,
                                tx % 32 + 32,
                            ]
                            h_lane_2[jv, tx] = h0_source[
                                state_idx,
                                bh,
                                bv * value_span + (tx // 32) * block_DV + jv,
                                tx % 32 + 64,
                            ]
                            h_lane_3[jv, tx] = h0_source[
                                state_idx,
                                bh,
                                bv * value_span + (tx // 32) * block_DV + jv,
                                tx % 32 + 96,
                            ]

            for step in T.serial(tokens_per_seq):
                token_idx = seq_start + step

                if step != 0 and cache_idx >= 0:
                    parent_step[0] = retrieve_parent_token[bn, step]
                    if parent_step[0] != step - 1:
                        for jv in T.serial(block_DV):
                            for tx in T.Parallel(64):
                                if bv * value_span + (tx // 32) * block_DV + jv < DV:
                                    h_lane_0[jv, tx] = intermediate_states_buffer[
                                        cache_idx,
                                        parent_step[0],
                                        bh,
                                        bv * value_span + (tx // 32) * block_DV + jv,
                                        tx % 32,
                                    ]
                                    h_lane_1[jv, tx] = intermediate_states_buffer[
                                        cache_idx,
                                        parent_step[0],
                                        bh,
                                        bv * value_span + (tx // 32) * block_DV + jv,
                                        tx % 32 + 32,
                                    ]
                                    h_lane_2[jv, tx] = intermediate_states_buffer[
                                        cache_idx,
                                        parent_step[0],
                                        bh,
                                        bv * value_span + (tx // 32) * block_DV + jv,
                                        tx % 32 + 64,
                                    ]
                                    h_lane_3[jv, tx] = intermediate_states_buffer[
                                        cache_idx,
                                        parent_step[0],
                                        bh,
                                        bv * value_span + (tx // 32) * block_DV + jv,
                                        tx % 32 + 96,
                                    ]

                for tx in T.Parallel(64):
                    q_lane_0[tx] = q[0, token_idx, bhq, tx % 32]
                    q_lane_1[tx] = q[0, token_idx, bhq, tx % 32 + 32]
                    q_lane_2[tx] = q[0, token_idx, bhq, tx % 32 + 64]
                    q_lane_3[tx] = q[0, token_idx, bhq, tx % 32 + 96]
                    k_lane_0[tx] = k[0, token_idx, bhq, tx % 32]
                    k_lane_1[tx] = k[0, token_idx, bhq, tx % 32 + 32]
                    k_lane_2[tx] = k[0, token_idx, bhq, tx % 32 + 64]
                    k_lane_3[tx] = k[0, token_idx, bhq, tx % 32 + 96]
                    q_norm[tx] = (
                        q_lane_0[tx] * q_lane_0[tx]
                        + q_lane_1[tx] * q_lane_1[tx]
                        + q_lane_2[tx] * q_lane_2[tx]
                        + q_lane_3[tx] * q_lane_3[tx]
                    )
                    k_norm[tx] = (
                        k_lane_0[tx] * k_lane_0[tx]
                        + k_lane_1[tx] * k_lane_1[tx]
                        + k_lane_2[tx] * k_lane_2[tx]
                        + k_lane_3[tx] * k_lane_3[tx]
                    )

                    if use_qk_l2norm_in_kernel:
                        q_norm[tx] = T.warp_reduce_sum(q_norm[tx])
                        k_norm[tx] = T.warp_reduce_sum(k_norm[tx])
                        q_norm[tx] = T.rsqrt(q_norm[tx] + 1e-6)
                        k_norm[tx] = T.rsqrt(k_norm[tx] + 1e-6)
                        q_lane_0[tx] *= q_norm[tx]
                        q_lane_1[tx] *= q_norm[tx]
                        q_lane_2[tx] *= q_norm[tx]
                        q_lane_3[tx] *= q_norm[tx]
                        k_lane_0[tx] *= k_norm[tx]
                        k_lane_1[tx] *= k_norm[tx]
                        k_lane_2[tx] *= k_norm[tx]
                        k_lane_3[tx] *= k_norm[tx]

                    q_lane_0[tx] *= scale
                    q_lane_1[tx] *= scale
                    q_lane_2[tx] *= scale
                    q_lane_3[tx] *= scale

                    a_raw[tx] = a[0, token_idx, bh]
                    b_raw[tx] = b[0, token_idx, bh]
                    x[tx] = a_raw[tx] + dt_bias[bh]
                    beta_x[tx] = softplus_beta * x[tx]
                    if beta_x[tx] <= softplus_threshold:
                        softplus_x[tx] = (
                            T.log(1.0 + T.exp(beta_x[tx])) / softplus_beta
                        )
                    else:
                        softplus_x[tx] = x[tx]
                    exp_g[tx] = T.exp(-T.exp(A_log[bh]) * softplus_x[tx])
                    beta[tx] = 1.0 / (1.0 + T.exp(-b_raw[tx]))

                for jv in T.serial(block_DV):
                    for tx in T.Parallel(64):
                        if bv * value_span + (tx // 32) * block_DV + jv < DV:
                            h_lane_0[jv, tx] *= exp_g[tx]
                            h_lane_1[jv, tx] *= exp_g[tx]
                            h_lane_2[jv, tx] *= exp_g[tx]
                            h_lane_3[jv, tx] *= exp_g[tx]
                            local_sum[tx] = (
                                h_lane_0[jv, tx] * k_lane_0[tx]
                                + h_lane_1[jv, tx] * k_lane_1[tx]
                                + h_lane_2[jv, tx] * k_lane_2[tx]
                                + h_lane_3[jv, tx] * k_lane_3[tx]
                            )

                            kv_value[tx] = T.warp_reduce_sum(local_sum[tx])
                            v_delta[tx] = (
                                v[
                                    0,
                                    token_idx,
                                    bh,
                                    bv * value_span + (tx // 32) * block_DV + jv,
                                ]
                                - kv_value[tx]
                            ) * beta[tx]

                            h_lane_0[jv, tx] += k_lane_0[tx] * v_delta[tx]
                            h_lane_1[jv, tx] += k_lane_1[tx] * v_delta[tx]
                            h_lane_2[jv, tx] += k_lane_2[tx] * v_delta[tx]
                            h_lane_3[jv, tx] += k_lane_3[tx] * v_delta[tx]
                            local_sum[tx] = (
                                h_lane_0[jv, tx] * q_lane_0[tx]
                                + h_lane_1[jv, tx] * q_lane_1[tx]
                                + h_lane_2[jv, tx] * q_lane_2[tx]
                                + h_lane_3[jv, tx] * q_lane_3[tx]
                            )

                            o_value[tx] = T.warp_reduce_sum(local_sum[tx])
                            if tx % 32 == 0:
                                o[
                                    0,
                                    token_idx,
                                    bh,
                                    bv * value_span + (tx // 32) * block_DV + jv,
                                ] = o_value[tx]

                            if cache_idx >= 0:
                                intermediate_states_buffer[
                                    cache_idx,
                                    step,
                                    bh,
                                    bv * value_span + (tx // 32) * block_DV + jv,
                                    tx % 32,
                                ] = h_lane_0[jv, tx]
                                intermediate_states_buffer[
                                    cache_idx,
                                    step,
                                    bh,
                                    bv * value_span + (tx // 32) * block_DV + jv,
                                    tx % 32 + 32,
                                ] = h_lane_1[jv, tx]
                                intermediate_states_buffer[
                                    cache_idx,
                                    step,
                                    bh,
                                    bv * value_span + (tx // 32) * block_DV + jv,
                                    tx % 32 + 64,
                                ] = h_lane_2[jv, tx]
                                intermediate_states_buffer[
                                    cache_idx,
                                    step,
                                    bh,
                                    bv * value_span + (tx // 32) * block_DV + jv,
                                    tx % 32 + 96,
                                ] = h_lane_3[jv, tx]

    return tilelang_flashqla_gdn_tree_verify_bv8x2_warp_kernel


def _identity_i32(n: int, device: torch.device):
    device = torch.device(device)
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    key = (index, n)
    cached = _IDENTITY_I32_CACHE.get(key)
    if cached is None or cached.device.index != index:
        cached = torch.arange(n, dtype=torch.int32, device=device)
        _IDENTITY_I32_CACHE[key] = cached
    return cached


def _supports_regular_recurrent_kernel(device: torch.device) -> bool:
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


def _contiguous_state_view(state: torch.Tensor, num_value_heads: int, head_k_dim: int, head_v_dim: int):
    if state.dim() != 4:
        raise ValueError("initial_state_source must be a 4-D tensor")
    if state.shape[1] != num_value_heads:
        raise ValueError(
            f"initial_state_source head dim {state.shape[1]} != num_value_heads {num_value_heads}"
        )
    if state.shape[2:] == (head_v_dim, head_k_dim):
        return state.contiguous()
    if state.shape[2:] == (head_k_dim, head_v_dim):
        return state.contiguous().view(state.shape[0], num_value_heads, head_v_dim, head_k_dim)
    raise ValueError(
        "initial_state_source must have trailing shape [V, K] for recurrent GDN layout "
        "or [K, V] when K == V-compatible contiguous storage is intended"
    )


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor | None = None,
    scale: float | None = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: torch.Tensor | None = None,
    intermediate_state_indices: torch.Tensor | None = None,
    cache_steps: int | None = None,
    retrieve_parent_token: torch.Tensor | None = None,
    block_dv: int | None = None,
    assume_regular: bool = False,
    b_is_beta: bool = False,
    a_is_log_decay: bool = False,
) -> torch.Tensor:
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("FlashQLA recurrent GDN kernel expects flattened [1, total_tokens, ...] inputs")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not q.is_cuda:
        raise ValueError("FlashQLA recurrent GDN kernel requires CUDA tensors")
    if (
        assume_regular
        and initial_state_indices is None
        and block_dv is None
        and intermediate_states_buffer is None
        and intermediate_state_indices is None
        and cache_steps is None
        and retrieve_parent_token is None
        and not disable_state_update
        and not _autotune_enabled()
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and a.is_contiguous()
        and b.is_contiguous()
        and initial_state_source.is_contiguous()
    ):
        _, total_tokens_fast, num_key_heads_fast, head_k_dim_fast = q.shape
        _, _, num_value_heads_fast, head_v_dim_fast = v.shape
        if (
            total_tokens_fast in (1, 2, 4)
            and initial_state_source.dim() == 4
            and initial_state_source.shape
            == (
                total_tokens_fast,
                num_value_heads_fast,
                head_v_dim_fast,
                head_k_dim_fast,
            )
            and num_key_heads_fast == 16
            and num_value_heads_fast == 48
            and head_k_dim_fast == 128
            and head_v_dim_fast == 128
            and num_value_heads_fast % num_key_heads_fast == 0
            and a.shape == (1, total_tokens_fast, num_value_heads_fast)
            and b.shape == (1, total_tokens_fast, num_value_heads_fast)
            and A_log.shape == (num_value_heads_fast,)
            and dt_bias.shape == (num_value_heads_fast,)
        ):
            scale_fast = head_k_dim_fast ** -0.5 if scale is None else scale
            A_log_c = A_log if A_log.is_contiguous() else A_log.contiguous()
            dt_bias_c = dt_bias if dt_bias.is_contiguous() else dt_bias.contiguous()
            o = torch.empty_like(v)
            if (
                total_tokens_fast == 2
                and a_is_log_decay
                and b_is_beta
                and use_qk_l2norm_in_kernel
            ):
                kernel_key = (
                    torch.cuda.current_device(),
                    "qwen_decode_precomputed_qknorm_static_b2",
                    scale_fast,
                    q.dtype,
                    k.dtype,
                    v.dtype,
                    a.dtype,
                    b.dtype,
                    initial_state_source.dtype,
                    o.dtype,
                )
                kernel = _RECURRENT_KERNEL_CACHE.get(kernel_key)
                if kernel is None:
                    kernel = tilelang_flashqla_gdn_decode_precomputed_qknorm_bv16_warp(
                        H=num_key_heads_fast,
                        HV=num_value_heads_fast,
                        DK=head_k_dim_fast,
                        DV=head_v_dim_fast,
                        scale=scale_fast,
                        q_dtype=q.dtype,
                        k_dtype=k.dtype,
                        v_dtype=v.dtype,
                        g_dtype=a.dtype,
                        beta_dtype=b.dtype,
                        state_dtype=initial_state_source.dtype,
                        o_dtype=o.dtype,
                        accum_dtype="float32",
                        block_DV=4,
                        max_nreg=0,
                        static_B=2,
                    )
                    _RECURRENT_KERNEL_CACHE[kernel_key] = kernel
                kernel(
                    a,
                    q,
                    k,
                    v,
                    b,
                    initial_state_source,
                    o,
                )
                return o
            static_block_dv_fast = 16 if total_tokens_fast == 4 else 4
            static_max_nreg_fast = 72 if total_tokens_fast == 2 else 128
            kernel_key = (
                torch.cuda.current_device(),
                "qwen_decode_static_b",
                total_tokens_fast,
                static_block_dv_fast,
                static_max_nreg_fast,
                scale_fast,
                q.dtype,
                k.dtype,
                v.dtype,
                a.dtype,
                b.dtype,
                A_log.dtype,
                dt_bias.dtype,
                initial_state_source.dtype,
                o.dtype,
                use_qk_l2norm_in_kernel,
                b_is_beta,
                a_is_log_decay,
            )
            kernel = _RECURRENT_KERNEL_CACHE.get(kernel_key)
            if kernel is None:
                kernel = tilelang_flashqla_gdn_decode_b4_bv16_warp(
                    H=num_key_heads_fast,
                    HV=num_value_heads_fast,
                    DK=head_k_dim_fast,
                    DV=head_v_dim_fast,
                    scale=scale_fast,
                    softplus_beta=softplus_beta,
                    softplus_threshold=softplus_threshold,
                    q_dtype=q.dtype,
                    k_dtype=k.dtype,
                    v_dtype=v.dtype,
                    a_dtype=a.dtype,
                    b_dtype=b.dtype,
                    a_log_dtype=A_log.dtype,
                    dt_bias_dtype=dt_bias.dtype,
                    state_dtype=initial_state_source.dtype,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    block_DV=static_block_dv_fast,
                    max_nreg=static_max_nreg_fast,
                    b_is_beta=b_is_beta,
                    a_is_log_decay=a_is_log_decay,
                    static_B=total_tokens_fast,
                )
                _RECURRENT_KERNEL_CACHE[kernel_key] = kernel
            kernel(
                A_log_c,
                a,
                dt_bias_c,
                q,
                k,
                v,
                b,
                initial_state_source,
                o,
            )
            return o
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
    if not (a.is_contiguous() and b.is_contiguous()):
        a = a.contiguous()
        b = b.contiguous()

    _, total_tokens, num_key_heads, head_k_dim = q.shape
    _, _, num_value_heads, head_v_dim = v.shape
    if num_value_heads % num_key_heads != 0:
        raise ValueError("num_value_heads must be divisible by num_key_heads")
    if head_k_dim != 128 or head_v_dim != 128:
        raise ValueError("FlashQLA recurrent GDN kernel currently specializes Qwen GDN K=V=128")
    if a.shape != (1, total_tokens, num_value_heads):
        raise ValueError(f"a shape {tuple(a.shape)} does not match [1, total_tokens, HV]")
    if b.shape != (1, total_tokens, num_value_heads):
        raise ValueError(f"b shape {tuple(b.shape)} does not match [1, total_tokens, HV]")
    if A_log.shape != (num_value_heads,):
        raise ValueError("A_log must have shape [HV]")
    if dt_bias.shape != (num_value_heads,):
        raise ValueError("dt_bias must have shape [HV]")
    A_log_c = A_log if A_log.is_contiguous() else A_log.contiguous()
    dt_bias_c = dt_bias if dt_bias.is_contiguous() else dt_bias.contiguous()

    state = _contiguous_state_view(
        initial_state_source, num_value_heads, head_k_dim, head_v_dim
    )
    num_sequences = state.shape[0]
    use_regular_kernel = _supports_regular_recurrent_kernel(q.device) and (
        assume_regular or cu_seqlens is None
    )
    if total_tokens % num_sequences != 0:
        if use_regular_kernel and assume_regular:
            raise ValueError("assume_regular=True requires total_tokens divisible by num_sequences")
        regular_tokens_per_seq = None
    else:
        regular_tokens_per_seq = total_tokens // num_sequences
    if scale is None:
        scale = head_k_dim ** -0.5
    initial_state_indices_is_identity = initial_state_indices is None
    if (
        use_regular_kernel
        and assume_regular
        and regular_tokens_per_seq == 1
        and intermediate_states_buffer is None
        and retrieve_parent_token is None
        and not disable_state_update
        and not _autotune_enabled()
    ):
        if initial_state_indices is None:
            initial_state_indices = _identity_i32(num_sequences, q.device)
        else:
            initial_state_indices = initial_state_indices.to(
                device=q.device, dtype=torch.int32
            ).contiguous()
        identity_indices = initial_state_indices_is_identity
        if block_dv is not None:
            block_DV = block_dv
        elif num_sequences == 1:
            block_DV = 4
        elif num_sequences == 2:
            block_DV = 4
        elif 3 <= num_sequences <= 4:
            block_DV = 16
        elif 5 <= num_sequences < 16:
            block_DV = 8
        elif num_sequences == 16:
            block_DV = 2
        else:
            block_DV = 4
        if (
            block_DV in (32, head_v_dim)
            or (block_DV in (1, 2, 4, 8, 16) and num_sequences <= 16)
            or (block_DV in (2, 4) and num_sequences >= 16)
            or (block_DV == 8 and num_sequences <= 16)
            or (block_DV == 16 and num_sequences <= 4)
        ):
            use_gqa3_decode = False
            use_prepared_gate_decode = False
            use_stream_decode = False
            o = torch.empty_like(v)
            if (
                initial_state_indices_is_identity
                and block_dv is None
                and num_sequences == 1
                and num_key_heads == 16
                and num_value_heads == 48
                and block_DV == 4
            ):
                kernel = _get_recurrent_kernel(
                    tilelang_flashqla_gdn_decode_b4_bv16_warp,
                    H=num_key_heads,
                    HV=num_value_heads,
                    DK=head_k_dim,
                    DV=head_v_dim,
                    scale=scale,
                    softplus_beta=softplus_beta,
                    softplus_threshold=softplus_threshold,
                    q_dtype=q.dtype,
                    k_dtype=k.dtype,
                    v_dtype=v.dtype,
                    a_dtype=a.dtype,
                    b_dtype=b.dtype,
                    a_log_dtype=A_log.dtype,
                    dt_bias_dtype=dt_bias.dtype,
                    state_dtype=state.dtype,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    block_DV=4,
                    max_nreg=128,
                    b_is_beta=b_is_beta,
                    a_is_log_decay=a_is_log_decay,
                    static_B=1,
                )
                kernel(
                    A_log_c,
                    a,
                    dt_bias_c,
                    q,
                    k,
                    v,
                    b,
                    state,
                    o,
                )
                return o
            if (
                initial_state_indices_is_identity
                and block_dv is None
                and num_sequences == 2
                and num_key_heads == 16
                and num_value_heads == 48
                and block_DV == 8
                and a_is_log_decay
                and b_is_beta
                and use_gqa3_decode
            ):
                kernel = _get_recurrent_kernel(
                    tilelang_flashqla_gdn_decode_gqa3_bv16_warp,
                    H=num_key_heads,
                    HV=num_value_heads,
                    DK=head_k_dim,
                    DV=head_v_dim,
                    scale=scale,
                    softplus_beta=softplus_beta,
                    softplus_threshold=softplus_threshold,
                    q_dtype=q.dtype,
                    k_dtype=k.dtype,
                    v_dtype=v.dtype,
                    a_dtype=a.dtype,
                    b_dtype=b.dtype,
                    a_log_dtype=A_log.dtype,
                    dt_bias_dtype=dt_bias.dtype,
                    state_dtype=state.dtype,
                    indices_dtype=torch.int32,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    identity_indices=False,
                    b_is_beta=True,
                    a_is_log_decay=True,
                )
                kernel(
                    A_log_c,
                    a,
                    dt_bias_c,
                    q,
                    k,
                    v,
                    b,
                    state,
                    initial_state_indices,
                    o,
                )
                return o
            if (
                initial_state_indices_is_identity
                and block_dv is None
                and num_sequences == 2
                and num_key_heads == 16
                and num_value_heads == 48
                and block_DV == 4
            ):
                kernel = _get_recurrent_kernel(
                    tilelang_flashqla_gdn_decode_b4_bv16_warp,
                    H=num_key_heads,
                    HV=num_value_heads,
                    DK=head_k_dim,
                    DV=head_v_dim,
                    scale=scale,
                    softplus_beta=softplus_beta,
                    softplus_threshold=softplus_threshold,
                    q_dtype=q.dtype,
                    k_dtype=k.dtype,
                    v_dtype=v.dtype,
                    a_dtype=a.dtype,
                    b_dtype=b.dtype,
                    a_log_dtype=A_log.dtype,
                    dt_bias_dtype=dt_bias.dtype,
                    state_dtype=state.dtype,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    block_DV=4,
                    max_nreg=72,
                    b_is_beta=b_is_beta,
                    a_is_log_decay=a_is_log_decay,
                    static_B=2,
                )
                kernel(
                    A_log_c,
                    a,
                    dt_bias_c,
                    q,
                    k,
                    v,
                    b,
                    state,
                    o,
                )
                return o
            if (
                initial_state_indices_is_identity
                and block_dv is None
                and num_sequences == 4
                and num_key_heads == 16
                and num_value_heads == 48
                and block_DV == 16
            ):
                if (
                    a_is_log_decay
                    and b_is_beta
                    and not use_qk_l2norm_in_kernel
                ):
                    kernel = _get_recurrent_kernel(
                        tilelang_flashqla_gdn_decode_b4_precomputed_bv16_warp,
                        H=num_key_heads,
                        HV=num_value_heads,
                        DK=head_k_dim,
                        DV=head_v_dim,
                        scale=scale,
                        q_dtype=q.dtype,
                        k_dtype=k.dtype,
                        v_dtype=v.dtype,
                        g_dtype=a.dtype,
                        beta_dtype=b.dtype,
                        state_dtype=state.dtype,
                        o_dtype=o.dtype,
                        accum_dtype="float32",
                        max_nreg=120,
                    )
                    kernel(
                        a,
                        q,
                        k,
                        v,
                        b,
                        state,
                        o,
                    )
                    return o
                static_block_DV = 16 if a_is_log_decay else 11
                static_max_nreg = 128 if a_is_log_decay else 80
                kernel = _get_recurrent_kernel(
                    tilelang_flashqla_gdn_decode_b4_bv16_warp,
                    H=num_key_heads,
                    HV=num_value_heads,
                    DK=head_k_dim,
                    DV=head_v_dim,
                    scale=scale,
                    softplus_beta=softplus_beta,
                    softplus_threshold=softplus_threshold,
                    q_dtype=q.dtype,
                    k_dtype=k.dtype,
                    v_dtype=v.dtype,
                    a_dtype=a.dtype,
                    b_dtype=b.dtype,
                    a_log_dtype=A_log.dtype,
                    dt_bias_dtype=dt_bias.dtype,
                    state_dtype=state.dtype,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    block_DV=static_block_DV,
                    max_nreg=static_max_nreg,
                    b_is_beta=b_is_beta,
                    a_is_log_decay=a_is_log_decay,
                    static_B=4,
                )
                kernel(
                    A_log_c,
                    a,
                    dt_bias_c,
                    q,
                    k,
                    v,
                    b,
                    state,
                    o,
                )
                return o
            if (
                initial_state_indices_is_identity
                and block_dv is None
                and num_sequences == 16
                and num_key_heads == 16
                and num_value_heads == 48
                and a_is_log_decay
                and b_is_beta
                and use_qk_l2norm_in_kernel
            ):
                kernel = _get_recurrent_kernel(
                    tilelang_flashqla_gdn_decode_bv16_split4_warp,
                    H=num_key_heads,
                    HV=num_value_heads,
                    DK=head_k_dim,
                    DV=head_v_dim,
                    scale=scale,
                    softplus_beta=softplus_beta,
                    softplus_threshold=softplus_threshold,
                    q_dtype=q.dtype,
                    k_dtype=k.dtype,
                    v_dtype=v.dtype,
                    a_dtype=a.dtype,
                    b_dtype=b.dtype,
                    a_log_dtype=A_log.dtype,
                    dt_bias_dtype=dt_bias.dtype,
                    state_dtype=state.dtype,
                    indices_dtype=torch.int32,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_qk_l2norm_in_kernel=True,
                    identity_indices=True,
                    max_nreg=160,
                    b_is_beta=True,
                    a_is_log_decay=True,
                )
                kernel(
                    A_log_c,
                    a,
                    dt_bias_c,
                    q,
                    k,
                    v,
                    b,
                    state,
                    initial_state_indices,
                    o,
                )
                return o
            if use_stream_decode:
                kernel = _get_recurrent_kernel(
                    tilelang_flashqla_gdn_decode_bv16_stream_warp,
                    H=num_key_heads,
                    HV=num_value_heads,
                    DK=head_k_dim,
                    DV=head_v_dim,
                    scale=scale,
                    softplus_beta=softplus_beta,
                    softplus_threshold=softplus_threshold,
                    q_dtype=q.dtype,
                    k_dtype=k.dtype,
                    v_dtype=v.dtype,
                    a_dtype=a.dtype,
                    b_dtype=b.dtype,
                    a_log_dtype=A_log.dtype,
                    dt_bias_dtype=dt_bias.dtype,
                    state_dtype=state.dtype,
                    indices_dtype=torch.int32,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    identity_indices=identity_indices,
                )
                kernel(
                    A_log_c,
                    a,
                    dt_bias_c,
                    q,
                    k,
                    v,
                    b,
                    state,
                    initial_state_indices,
                    o,
                )
                return o
            if use_gqa3_decode:
                kernel = _get_recurrent_kernel(
                    tilelang_flashqla_gdn_decode_gqa3_bv16_warp,
                    H=num_key_heads,
                    HV=num_value_heads,
                    DK=head_k_dim,
                    DV=head_v_dim,
                    scale=scale,
                    softplus_beta=softplus_beta,
                    softplus_threshold=softplus_threshold,
                    q_dtype=q.dtype,
                    k_dtype=k.dtype,
                    v_dtype=v.dtype,
                    a_dtype=a.dtype,
                    b_dtype=b.dtype,
                    a_log_dtype=A_log.dtype,
                    dt_bias_dtype=dt_bias.dtype,
                    state_dtype=state.dtype,
                    indices_dtype=torch.int32,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    identity_indices=identity_indices,
                )
                kernel(
                    A_log_c,
                    a,
                    dt_bias_c,
                    q,
                    k,
                    v,
                    b,
                    state,
                    initial_state_indices,
                    o,
                )
                return o
            if use_prepared_gate_decode:
                exp_g_pre, beta_pre = _prepare_decode_gates(
                    A_log=A_log,
                    a=a,
                    dt_bias=dt_bias,
                    b=b,
                    softplus_beta=softplus_beta,
                    softplus_threshold=softplus_threshold,
                )
                kernel = _get_recurrent_kernel(
                    tilelang_flashqla_gdn_decode_bv16_prepared_gate_warp,
                    H=num_key_heads,
                    HV=num_value_heads,
                    DK=head_k_dim,
                    DV=head_v_dim,
                    scale=scale,
                    q_dtype=q.dtype,
                    k_dtype=k.dtype,
                    v_dtype=v.dtype,
                    gate_dtype=exp_g_pre.dtype,
                    state_dtype=state.dtype,
                    indices_dtype=torch.int32,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    block_DV=block_DV,
                    identity_indices=True,
                )
                kernel(
                    exp_g_pre,
                    beta_pre,
                    q,
                    k,
                    v,
                    state,
                    initial_state_indices,
                    o,
                )
                return o
            if (
                (block_DV in (2, 4) and num_sequences >= 16)
                or (block_DV in (8, 16) and num_sequences <= 16)
            ):
                decode_kernel = tilelang_flashqla_gdn_decode_bv16_regular_warp
                decode_kwargs = {
                    "block_DV": block_DV,
                    "identity_indices": identity_indices,
                    "max_nreg": (
                        80
                        if (block_DV == 4 and num_sequences == 1)
                        else 128
                        if (block_DV == 4 and num_sequences == 2)
                        else 112
                        if (block_DV == 16 and num_sequences == 4)
                        else 64
                        if (block_DV == 1 and 5 <= num_sequences <= 16)
                        else 0
                    ),
                    "b_is_beta": b_is_beta,
                    "a_is_log_decay": a_is_log_decay,
                }
            else:
                if block_DV == 32:
                    decode_kernel = (
                        tilelang_flashqla_gdn_decode_bv32x2_warp
                        if num_sequences <= 32
                        else tilelang_flashqla_gdn_decode_bv32_warp
                    )
                    decode_kwargs = {}
                else:
                    decode_kernel = tilelang_flashqla_gdn_decode_fullv
                    decode_kwargs = {
                        "b_is_beta": b_is_beta,
                        "a_is_log_decay": a_is_log_decay,
                    }
            kernel = _get_recurrent_kernel(
                decode_kernel,
                H=num_key_heads,
                HV=num_value_heads,
                DK=head_k_dim,
                DV=head_v_dim,
                scale=scale,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                v_dtype=v.dtype,
                a_dtype=a.dtype,
                b_dtype=b.dtype,
                a_log_dtype=A_log.dtype,
                dt_bias_dtype=dt_bias.dtype,
                state_dtype=state.dtype,
                indices_dtype=torch.int32,
                o_dtype=o.dtype,
                accum_dtype="float32",
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                **decode_kwargs,
            )
            kernel(
                A_log_c,
                a,
                dt_bias_c,
                q,
                k,
                v,
                b,
                state,
                initial_state_indices,
                o,
            )
            return o
    if cu_seqlens is None:
        if regular_tokens_per_seq is None:
            raise ValueError("total_tokens must be divisible by num_sequences when cu_seqlens is omitted")
        cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            regular_tokens_per_seq,
            dtype=torch.int32,
            device=q.device,
        )
    else:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        if cu_seqlens.numel() != num_sequences + 1:
            raise ValueError("cu_seqlens length must equal num_sequences + 1")
    identity_indices = False
    if initial_state_indices is None:
        initial_state_indices = _identity_i32(num_sequences, q.device)
    else:
        initial_state_indices = initial_state_indices.to(device=q.device, dtype=torch.int32).contiguous()

    cache_intermediate = intermediate_states_buffer is not None
    if cache_intermediate:
        if cache_steps is None:
            cache_steps = intermediate_states_buffer.shape[1]
        intermediate = _contiguous_state_view(
            intermediate_states_buffer.view(-1, num_value_heads, head_v_dim, head_k_dim),
            num_value_heads,
            head_k_dim,
            head_v_dim,
        ).view(intermediate_states_buffer.shape[0], intermediate_states_buffer.shape[1], num_value_heads, head_v_dim, head_k_dim)
        if intermediate.shape[0] < num_sequences:
            raise ValueError("intermediate_states_buffer must have at least num_sequences slots")
        if intermediate.shape[1] < cache_steps:
            raise ValueError("intermediate_states_buffer cache dimension is smaller than cache_steps")
        if intermediate_state_indices is None:
            intermediate_state_indices = _identity_i32(num_sequences, q.device)
        else:
            intermediate_state_indices = intermediate_state_indices.to(
                device=q.device, dtype=torch.int32
            ).contiguous()
    else:
        cache_steps = 1
        intermediate = None
        intermediate_state_indices = initial_state_indices

    has_tree = retrieve_parent_token is not None
    if has_tree:
        retrieve_parent_token = retrieve_parent_token.to(device=q.device, dtype=torch.int32).contiguous()
        if retrieve_parent_token.shape[0] != num_sequences:
            raise ValueError("retrieve_parent_token first dimension must equal num_sequences")
    else:
        retrieve_parent_token = None

    o = torch.empty_like(v)
    if block_dv is not None:
        block_DV = block_dv
    elif (
        use_regular_kernel
        and regular_tokens_per_seq == 1
        and not has_tree
        and not cache_intermediate
        and not disable_state_update
        and num_sequences == 1
    ):
        block_DV = head_v_dim
    elif (
        use_regular_kernel
        and regular_tokens_per_seq == 1
        and not has_tree
        and not cache_intermediate
        and not disable_state_update
        and 5 <= num_sequences < 16
    ):
        block_DV = 8
    elif (
        use_regular_kernel
        and regular_tokens_per_seq == 1
        and not has_tree
        and not cache_intermediate
        and not disable_state_update
        and 2 <= num_sequences <= 4
    ):
        block_DV = 16
    elif (
        use_regular_kernel
        and regular_tokens_per_seq == 1
        and not has_tree
        and not cache_intermediate
        and not disable_state_update
        and num_sequences == 16
    ):
        block_DV = 2
    elif (
        use_regular_kernel
        and regular_tokens_per_seq == 1
        and not has_tree
        and not cache_intermediate
        and not disable_state_update
        and num_sequences > 16
    ):
        block_DV = 4
    elif (
        use_regular_kernel
        and has_tree
        and cache_intermediate
        and regular_tokens_per_seq is not None
        and regular_tokens_per_seq <= 32
        and num_sequences <= 8
    ):
        block_DV = 4
    elif (
        use_regular_kernel
        and cache_intermediate
        and not has_tree
        and disable_state_update
        and num_value_heads == 48
        and regular_tokens_per_seq is not None
        and regular_tokens_per_seq <= 32
        and num_sequences <= 8
    ):
        block_DV = 4 if num_sequences >= 4 or regular_tokens_per_seq == 2 else 8
    elif (
        use_regular_kernel
        and cache_intermediate
        and regular_tokens_per_seq is not None
        and regular_tokens_per_seq <= 16
        and num_sequences <= 4
    ):
        block_DV = 4
    elif (
        use_regular_kernel
        and cache_intermediate
        and regular_tokens_per_seq is not None
        and regular_tokens_per_seq <= 32
        and (num_sequences <= 2 or (has_tree and num_sequences <= 4))
    ):
        block_DV = 4
    elif use_regular_kernel and cache_intermediate:
        block_DV = 4 if has_tree and num_sequences == 4 else 8
    elif (
        use_regular_kernel
        and regular_tokens_per_seq == 1
        and not has_tree
        and not disable_state_update
    ):
        block_DV = 32
    else:
        block_DV = min(128, 1 << (head_v_dim - 1).bit_length())
    if block_dv is None and use_regular_kernel and _autotune_enabled():
        can_autotune = cache_intermediate or (
            regular_tokens_per_seq == 1
            and not has_tree
            and not disable_state_update
        )
        if can_autotune:
            cache_key = _autotune_cache_key(
                q,
                state,
                num_key_heads,
                num_value_heads,
                head_k_dim,
                head_v_dim,
                num_sequences,
                regular_tokens_per_seq,
                cache_steps,
                cache_intermediate,
                has_tree,
                disable_state_update,
                use_qk_l2norm_in_kernel,
            )
            cached_block_DV = _RECURRENT_AUTOTUNE_CACHE.get(cache_key)
            if cached_block_DV is not None:
                block_DV = cached_block_DV
            else:
                candidates = _autotune_block_dv_candidates(
                    num_sequences,
                    regular_tokens_per_seq,
                    head_v_dim,
                    cache_intermediate,
                    has_tree,
                    disable_state_update,
                )
                if len(candidates) > 1:
                    block_DV = _select_autotuned_block_dv(
                        block_DV,
                        candidates,
                        A_log,
                        a,
                        dt_bias,
                        q,
                        k,
                        v,
                        b,
                        state,
                        initial_state_indices,
                        scale,
                        softplus_beta,
                        softplus_threshold,
                        use_qk_l2norm_in_kernel,
                        cu_seqlens,
                        disable_state_update,
                        intermediate,
                        intermediate_state_indices,
                        cache_steps,
                        retrieve_parent_token,
                        cache_key,
                    )
    if (
        use_regular_kernel
        and regular_tokens_per_seq == 1
        and (
            block_DV in (32, head_v_dim)
            or (block_DV in (2, 4) and num_sequences >= 16)
            or (block_DV == 8 and num_sequences <= 16)
            or (block_DV == 16 and num_sequences <= 4)
        )
        and not cache_intermediate
        and not has_tree
        and not disable_state_update
    ):
        if (
            (block_DV in (2, 4) and num_sequences >= 16)
            or (block_DV in (8, 16) and num_sequences <= 16)
        ):
            decode_kernel = tilelang_flashqla_gdn_decode_bv16_regular_warp
            decode_kwargs = {
                "block_DV": block_DV,
                "identity_indices": identity_indices,
                "max_nreg": (
                    128
                    if (block_DV == 8 and num_sequences == 2)
                    else 96 if (block_DV == 16 and num_sequences == 4) else 0
                ),
                "b_is_beta": b_is_beta,
                "a_is_log_decay": a_is_log_decay,
            }
        else:
            if block_DV == 32:
                decode_kernel = (
                    tilelang_flashqla_gdn_decode_bv32x2_warp
                    if num_sequences <= 32
                    else tilelang_flashqla_gdn_decode_bv32_warp
                )
                decode_kwargs = {}
            else:
                decode_kernel = tilelang_flashqla_gdn_decode_fullv
                decode_kwargs = {
                    "b_is_beta": b_is_beta,
                    "a_is_log_decay": a_is_log_decay,
                }
        kernel = _get_recurrent_kernel(
            decode_kernel,
            H=num_key_heads,
            HV=num_value_heads,
            DK=head_k_dim,
            DV=head_v_dim,
            scale=scale,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            v_dtype=v.dtype,
            a_dtype=a.dtype,
            b_dtype=b.dtype,
            a_log_dtype=A_log.dtype,
            dt_bias_dtype=dt_bias.dtype,
            state_dtype=state.dtype,
            indices_dtype=torch.int32,
            o_dtype=o.dtype,
            accum_dtype="float32",
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **decode_kwargs,
        )
        kernel(
            A_log_c,
            a,
            dt_bias_c,
            q,
            k,
            v,
            b,
            state,
            initial_state_indices,
            o,
        )
        return o
    if intermediate is None:
        if cache_steps == 1:
            intermediate = state[:, None]
        else:
            intermediate = torch.empty(
                (num_sequences, 1, num_value_heads, head_v_dim, head_k_dim),
                dtype=state.dtype,
                device=q.device,
            )
    if retrieve_parent_token is None:
        if cache_steps == 1:
            retrieve_parent_token = initial_state_indices[:, None]
        else:
            retrieve_parent_token = torch.empty(
                (num_sequences, cache_steps), dtype=torch.int32, device=q.device
            )
    if (
        use_regular_kernel
        and has_tree
        and cache_intermediate
        and disable_state_update
        and block_DV == 8
        and num_sequences == 2
        and regular_tokens_per_seq is not None
        and regular_tokens_per_seq <= 16
    ):
        kernel = tilelang_flashqla_gdn_tree_verify_bv8x2_warp(
            H=num_key_heads,
            HV=num_value_heads,
            DK=head_k_dim,
            DV=head_v_dim,
            tokens_per_seq=regular_tokens_per_seq,
            scale=scale,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            v_dtype=v.dtype,
            a_dtype=a.dtype,
            b_dtype=b.dtype,
            a_log_dtype=A_log.dtype,
            dt_bias_dtype=dt_bias.dtype,
            state_dtype=state.dtype,
            indices_dtype=torch.int32,
            o_dtype=o.dtype,
            accum_dtype="float32",
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            block_DV=8,
        )
        kernel(
            A_log_c,
            a,
            dt_bias_c,
            q,
            k,
            v,
            b,
            state,
            initial_state_indices,
            intermediate,
            intermediate_state_indices,
            retrieve_parent_token,
            o,
        )
        return o
    if use_regular_kernel:
        if regular_tokens_per_seq is None:
            raise ValueError("regular FlashQLA recurrent kernel requires equal sequence lengths")
        if block_DV in (4, 8, 16, 32, 64):
            kernel = tilelang_flashqla_gdn_regular_bv32_warp(
                H=num_key_heads,
                HV=num_value_heads,
                DK=head_k_dim,
                DV=head_v_dim,
                tokens_per_seq=regular_tokens_per_seq,
                scale=scale,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                v_dtype=v.dtype,
                a_dtype=a.dtype,
                b_dtype=b.dtype,
                a_log_dtype=A_log.dtype,
                dt_bias_dtype=dt_bias.dtype,
                state_dtype=state.dtype,
                indices_dtype=torch.int32,
                o_dtype=o.dtype,
                accum_dtype="float32",
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                disable_state_update=disable_state_update,
                cache_intermediate_states=cache_intermediate,
                has_tree_attention=has_tree,
                block_DV=block_DV,
            )
            kernel(
                A_log_c,
                a,
                dt_bias_c,
                q,
                k,
                v,
                b,
                state,
                initial_state_indices,
                intermediate,
                intermediate_state_indices,
                retrieve_parent_token,
                o,
            )
        elif block_DV == head_v_dim:
            kernel = tilelang_flashqla_gdn_regular_fullv(
                H=num_key_heads,
                HV=num_value_heads,
                DK=head_k_dim,
                DV=head_v_dim,
                tokens_per_seq=regular_tokens_per_seq,
                scale=scale,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                v_dtype=v.dtype,
                a_dtype=a.dtype,
                b_dtype=b.dtype,
                a_log_dtype=A_log.dtype,
                dt_bias_dtype=dt_bias.dtype,
                state_dtype=state.dtype,
                indices_dtype=torch.int32,
                o_dtype=o.dtype,
                accum_dtype="float32",
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                disable_state_update=disable_state_update,
                cache_intermediate_states=cache_intermediate,
                has_tree_attention=has_tree,
            )
            kernel(
                A_log_c,
                a,
                dt_bias_c,
                q,
                k,
                v,
                b,
                state,
                initial_state_indices,
                intermediate,
                intermediate_state_indices,
                retrieve_parent_token,
                o,
            )
        else:
            kernel = tilelang_flashqla_gdn_regular(
                H=num_key_heads,
                HV=num_value_heads,
                DK=head_k_dim,
                DV=head_v_dim,
                tokens_per_seq=regular_tokens_per_seq,
                scale=scale,
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                v_dtype=v.dtype,
                a_dtype=a.dtype,
                b_dtype=b.dtype,
                a_log_dtype=A_log.dtype,
                dt_bias_dtype=dt_bias.dtype,
                state_dtype=state.dtype,
                indices_dtype=torch.int32,
                o_dtype=o.dtype,
                accum_dtype="float32",
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                disable_state_update=disable_state_update,
                cache_intermediate_states=cache_intermediate,
                has_tree_attention=has_tree,
                block_DV=block_DV,
            )
            kernel(
                A_log_c,
                a,
                dt_bias_c,
                q,
                k,
                v,
                b,
                state,
                initial_state_indices,
                intermediate,
                intermediate_state_indices,
                retrieve_parent_token,
                o,
            )
    else:
        kernel = tilelang_flashqla_gdn_update(
            H=num_key_heads,
            HV=num_value_heads,
            DK=head_k_dim,
            DV=head_v_dim,
            scale=scale,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            v_dtype=v.dtype,
            a_dtype=a.dtype,
            b_dtype=b.dtype,
            a_log_dtype=A_log.dtype,
            dt_bias_dtype=dt_bias.dtype,
            state_dtype=state.dtype,
            indices_dtype=torch.int32,
            o_dtype=o.dtype,
            accum_dtype="float32",
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            disable_state_update=disable_state_update,
            cache_intermediate_states=cache_intermediate,
            has_tree_attention=has_tree,
            block_DV=block_DV,
        )
        kernel(
            A_log_c,
            a,
            dt_bias_c,
            q,
            k,
            v,
            b,
            state,
            initial_state_indices,
            cu_seqlens,
            intermediate,
            intermediate_state_indices,
            retrieve_parent_token,
            o,
        )
    return o


flashqla_recurrent_gdn_update = fused_sigmoid_gating_delta_rule_update
