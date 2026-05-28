#!/usr/bin/env python3
import argparse
import sys

import torch

sys.path.insert(0, "tests")

from fla.ops.gated_delta_rule.chunk import (  # noqa: E402
    chunk_gated_delta_rule_bwd as fla_bwd,
)
from fla.ops.gated_delta_rule.chunk import (  # noqa: E402
    chunk_gated_delta_rule_fwd as fla_fwd,
)
from flash_qla import (  # noqa: E402
    chunk_gated_delta_rule_bwd as qla_bwd,
)
from flash_qla import (  # noqa: E402
    chunk_gated_delta_rule_fwd as qla_fwd,
)
from flash_qla.utils import l2norm  # noqa: E402
from test_gdr import test_gated_delta_rule  # noqa: E402


def bench_cuda_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(iters, 1)


def make_inputs(tokens: int, h: int, hv: int, kdim: int, vdim: int):
    torch.manual_seed(20260526 + tokens)
    device = "cuda"
    dtype = torch.bfloat16
    q = l2norm(torch.randn(1, tokens, h, kdim, device=device, dtype=dtype))
    k = l2norm(torch.randn(1, tokens, h, kdim, device=device, dtype=dtype))
    v = torch.randn(1, tokens, hv, vdim, device=device, dtype=dtype)
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, tokens, hv, device=device, dtype=torch.float32)
    ) / 16
    beta = torch.randn(1, tokens, hv, device=device, dtype=torch.float32).sigmoid()
    h0 = torch.randn(1, hv, kdim, vdim, device=device, dtype=torch.float32)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0) / 8
    return q, k, v, g, beta, h0, do, dht


def run_accuracy() -> None:
    test_gated_delta_rule(
        batch_size=1,
        num_tokens=512,
        num_k_heads=16,
        num_v_heads=48,
        head_dim_k=128,
        head_dim_v=128,
        varlen=False,
        use_h0=True,
        data_dtype="bfloat16",
        ref_dtype="float32",
        check_accuracy=True,
        show_speedup=False,
        auto_cp=False,
        random_seed=20260526,
    )


def run_case(tokens: int, warmup: int, iters: int) -> None:
    q, k, v, g, beta, h0, do, dht = make_inputs(tokens, 16, 48, 128, 128)
    scale = 128**-0.5
    g_fla, _o_fla, a_fla, _s_fla, _, _ = fla_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=None,
    )
    g_qla, a_qla, _o_qla, h_qla, _s_qla, v_new_qla, w_qla = qla_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        output_h=True,
        output_v_new=True,
        output_w=True,
        auto_cp=False,
    )

    def fla_call():
        return fla_bwd(q, k, v, g_fla, beta, a_fla, scale, h0, do, dht, None)

    def qla_call():
        return qla_bwd(
            q,
            k,
            v,
            g_qla,
            beta,
            a_qla,
            do,
            dht,
            scale,
            h0,
            None,
            h=h_qla,
            v_new=v_new_qla,
            w=w_qla,
        )

    fla_out = fla_call()
    qla_out = qla_call()
    torch.cuda.synchronize()
    diffs = []
    for lhs, rhs in zip(fla_out[:6], qla_out):
        if lhs is None or rhs is None:
            diffs.append(float("nan"))
        else:
            diffs.append((lhs.float() - rhs.float()).abs().max().item())

    fla_ms = bench_cuda_ms(fla_call, warmup, iters)
    qla_ms = bench_cuda_ms(qla_call, warmup, iters)
    print(
        f"bwd T={tokens} FLA={fla_ms:.4f} ms FlashQLA={qla_ms:.4f} ms "
        f"speedup={fla_ms / qla_ms:.3f}x "
        f"diffs={[round(x, 4) for x in diffs]}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="2048,8192")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--skip-accuracy", action="store_true")
    args = parser.parse_args()

    print(f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
    if not args.skip_accuracy:
        run_accuracy()
    for token_text in args.tokens.split(","):
        if token_text:
            run_case(int(token_text), args.warmup, args.iters)


if __name__ == "__main__":
    main()
