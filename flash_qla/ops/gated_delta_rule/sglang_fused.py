import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_sglang_fused_gdn(
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
    def tilelang_sglang_fused_gdn_kernel(
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

    return tilelang_sglang_fused_gdn_kernel


def _identity_i32(n: int, device: torch.device):
    return torch.arange(n, dtype=torch.int32, device=device)


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
        "initial_state_source must have trailing shape [V, K] for SGLang layout "
        "or [K, V] when K == V-compatible contiguous storage is intended"
    )


@torch.compiler.disable
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
) -> torch.Tensor:
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("SGLang-compatible FlashQLA kernel expects flattened [1, total_tokens, ...] inputs")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not q.is_cuda:
        raise ValueError("FlashQLA SGLang fused kernel requires CUDA tensors")
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
        raise ValueError("FlashQLA SGLang fused kernel currently specializes Qwen GDN K=V=128")
    if a.shape != (1, total_tokens, num_value_heads):
        raise ValueError(f"a shape {tuple(a.shape)} does not match [1, total_tokens, HV]")
    if b.shape != (1, total_tokens, num_value_heads):
        raise ValueError(f"b shape {tuple(b.shape)} does not match [1, total_tokens, HV]")
    if A_log.shape != (num_value_heads,):
        raise ValueError("A_log must have shape [HV]")
    if dt_bias.shape != (num_value_heads,):
        raise ValueError("dt_bias must have shape [HV]")

    state = _contiguous_state_view(
        initial_state_source, num_value_heads, head_k_dim, head_v_dim
    )
    num_sequences = state.shape[0]
    if cu_seqlens is None:
        if total_tokens % num_sequences != 0:
            raise ValueError("total_tokens must be divisible by num_sequences when cu_seqlens is omitted")
        tokens_per_seq = total_tokens // num_sequences
        cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            tokens_per_seq,
            dtype=torch.int32,
            device=q.device,
        )
    else:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        if cu_seqlens.numel() != num_sequences + 1:
            raise ValueError("cu_seqlens length must equal num_sequences + 1")
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
        intermediate = torch.empty(
            (num_sequences, 1, num_value_heads, head_v_dim, head_k_dim),
            dtype=state.dtype,
            device=q.device,
        )
        intermediate_state_indices = _identity_i32(num_sequences, q.device)

    has_tree = retrieve_parent_token is not None
    if has_tree:
        retrieve_parent_token = retrieve_parent_token.to(device=q.device, dtype=torch.int32).contiguous()
        if retrieve_parent_token.shape[0] != num_sequences:
            raise ValueError("retrieve_parent_token first dimension must equal num_sequences")
    else:
        retrieve_parent_token = torch.empty(
            (num_sequences, cache_steps), dtype=torch.int32, device=q.device
        )

    if scale is None:
        scale = head_k_dim ** -0.5

    o = torch.empty_like(v)
    block_DV = block_dv or min(128, 1 << (head_v_dim - 1).bit_length())
    kernel = tilelang_sglang_fused_gdn(
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
        A_log.contiguous(),
        a,
        dt_bias.contiguous(),
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


sglang_fused_gdn_update = fused_sigmoid_gating_delta_rule_update
