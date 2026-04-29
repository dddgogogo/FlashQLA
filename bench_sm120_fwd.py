import argparse
import statistics

import torch
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_fwd

from flash_qla import chunk_gated_delta_rule_fwd as qla_fwd
from flash_qla.utils import l2norm


def bench(fn, args, warmup, iters):
    fn(*args)
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def make_inputs(batch, tokens, h_k, h_v, dim, seed):
    torch.manual_seed(seed)
    q = l2norm(
        torch.randn((batch, tokens, h_k, dim), device="cuda", dtype=torch.bfloat16)
    )
    k = l2norm(
        torch.randn((batch, tokens, h_k, dim), device="cuda", dtype=torch.bfloat16)
    )
    v = torch.randn((batch, tokens, h_v, dim), device="cuda", dtype=torch.bfloat16)
    g = (
        torch.nn.functional.logsigmoid(
            torch.randn((batch, tokens, h_v), device="cuda", dtype=torch.float32)
        )
        / 16
    )
    beta = torch.randn(
        (batch, tokens, h_v), device="cuda", dtype=torch.float32
    ).sigmoid()
    h0 = torch.randn((batch, h_v, dim, dim), device="cuda", dtype=torch.float32) / 8
    scale = dim**-0.5
    return q, k, v, g, beta, scale, h0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=str, default="2048,8192,32768")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--k-heads", type=int, default=0)
    parser.add_argument("--v-heads", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compare-no-cp", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    print(
        "gpu",
        torch.cuda.get_device_name(args.device),
        torch.cuda.get_device_capability(args.device),
    )
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    h_k = args.k_heads or args.heads
    h_v = args.v_heads or args.heads

    for tokens in [int(x) for x in args.tokens.split(",") if x]:
        q, k, v, g, beta, scale, h0 = make_inputs(
            args.batch, tokens, h_k, h_v, args.dim, 42 + tokens
        )

        def run_fla(q, k, v, g, beta, scale, h0):
            return fla_fwd(q, k, v, g, beta, scale, h0, True, None)

        def run_qla(q, k, v, g, beta, scale, h0):
            return qla_fwd(q, k, v, g, beta, scale, h0, None, True, False, True)

        fla_times = bench(run_fla, (q, k, v, g, beta, scale, h0), args.warmup, args.iters)
        qla_times = bench(run_qla, (q, k, v, g, beta, scale, h0), args.warmup, args.iters)
        fla_med = statistics.median(fla_times)
        qla_med = statistics.median(qla_times)
        no_cp_text = ""
        if args.compare_no_cp:
            def run_qla_no_cp(q, k, v, g, beta, scale, h0):
                return qla_fwd(q, k, v, g, beta, scale, h0, None, True, False, False)

            qla_no_cp_times = bench(
                run_qla_no_cp,
                (q, k, v, g, beta, scale, h0),
                args.warmup,
                args.iters,
            )
            qla_no_cp_med = statistics.median(qla_no_cp_times)
            no_cp_text = (
                f" QLA_no_cp_ms={qla_no_cp_med:.3f} "
                f"cp_vs_no_cp={qla_no_cp_med / qla_med:.3f}x"
            )
        print(
            f"T={tokens} Hk={h_k} Hv={h_v} B={args.batch} "
            f"FLA_ms={fla_med:.3f} QLA_ms={qla_med:.3f} "
            f"speedup={fla_med / qla_med:.3f}x "
            f"FLA_min={min(fla_times):.3f} QLA_min={min(qla_times):.3f}"
            f"{no_cp_text}"
        )
        del q, k, v, g, beta, h0
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
