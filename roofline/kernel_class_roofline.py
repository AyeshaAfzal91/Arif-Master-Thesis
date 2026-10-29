#!/usr/bin/env python3
"""Roofline of an LLM inference step, aggregated per kernel class, from Nsight Compute raw CSV exports."""
import argparse
import csv
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

plt.rcParams.update({"axes.unicode_minus": False})

CLASS_ORDER = ["GEMM", "Attention", "SSM / GDN", "Elementwise", "Other"]
CLASS_SHAPES = {"GEMM": "o", "Attention": "^", "SSM / GDN": "D", "Elementwise": "s", "Other": "X"}
CLASS_RULES = [   # first match wins, on the lower-cased kernel name
    # cuDNN fused attention is named ..._sdpa_..._flash_fprop_...; without "sdpa" it lands in Other
    ("Attention", r"flash|fmha|sdpa|mla|attn|attention|merge_attn|paged"),
    ("SSM / GDN", r"selective_scan|ssd|chunk_scan|chunk_state|state_passing|chunk_cumsum|bmm_chunk|"
                  r"causal_conv1d|mamba|gated_delta|delta_rule|recurrent|chunk_.*fwd|recompute_w_u|l2norm|gdn"),
    ("GEMM", r"nvjet|gemm|gemv|cutlass|xmma|cublas|splitk|sm90_|sm100_|matmul|wgmma"),
    ("Elementwise", r"elementwise|vectorized|rms_norm|layer_norm|silu|act_and_mul|rotary|reshape_and_cache|"
                    r"copy|cat|concat|fill|index|add|mul|scale|norm"),
]
# tensor-core FLOP counters: floating-point types, dense (no _sparsity_on, no Blackwell _realtime)
TENSOR = re.compile(r"sm__ops_path_tensor_src_(bf16|fp16|tf32|fp8|fp4|fp6|fp64)[a-z0-9_]*\.sum")
TENSOR_SKIP = re.compile(r"_sparsity_on|_realtime")
PLAIN_FMA = {"smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,
             "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2, "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum": 2,
             "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum": 2, "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": 4,
             "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum": 2}
DERIVED_FMA = {"derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2": 1,
               "derived__sm__sass_thread_inst_executed_op_hfma_pred_on_x4": 1,
               "derived__sm__sass_thread_inst_executed_op_dfma_pred_on_x2": 1}
COL = {"t": "gpu__time_duration.sum", "sm_hz": "sm__cycles_elapsed.avg.per_second",
       "dram_hz": "dram__cycles_elapsed.avg.per_second", "rd": "dram__bytes_read.sum",
       "wr": "dram__bytes_write.sum", "dram_rate": "dram__bytes.sum.per_second",
       "dram_pk": "dram__bytes.sum.peak_sustained", "bf16_pk": "sm__ops_path_tensor_src_bf16_dst_fp32.sum.peak_sustained",
       "l2": "lts__t_bytes.sum", "l1": "l1tex__t_bytes.sum",
       "sol_tensor": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
       "sol_dram": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
       "sol_l2": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
       "sol_l1": "l1tex__throughput.avg.pct_of_peak_sustained_active"}
PREFIX = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}
TIME = {"ns": 1e-9, "nsecond": 1e-9, "us": 1e-6, "usecond": 1e-6, "ms": 1e-3, "msecond": 1e-3, "s": 1.0, "second": 1.0}
TAGS = (re.compile(r"(?P<run>.+?)_(?P<phase>prefill|decode|verify\w*)_s\d+_bs\d+"),
        re.compile(r"(?P<run>.+?_s\d+_bs\d+)_(?P<phase>prefill|decode|verify\w*)"))


def unit_scale(u):
    """Factor to SI base units (byte, second, hertz) for ncu unit strings such as 'Kbyte/s', 'usecond'."""
    u = (u or "").strip().lower()
    if u in TIME:
        return TIME[u]
    if "/" in u:                                    # rate: <prefix>byte / <time>
        num, den = u.split("/", 1)
        m = re.fullmatch(r"([kmgt]?)byte", num)
        return PREFIX[m.group(1)] / TIME.get(den, 1.0) if m else 1.0
    m = re.fullmatch(r"([kmgt]?)(byte|hz)", u)
    return PREFIX[m.group(1)] if m else 1.0


def load(path):
    """Per-kernel FLOPs, bytes, time and class from one raw CSV page (skips ==PROF== preamble lines)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    h = next(i for i, r in enumerate(rows) if r and r[0] in ("ID", "\"ID\""))
    cols, units = rows[h], rows[h + 1]
    df = pd.DataFrame(rows[h + 2:], columns=cols)
    num = lambda c: pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce").fillna(0.0).values \
        * unit_scale(units[cols.index(c)])
    out = pd.DataFrame({"name": df["Kernel Name"].astype(str).values})
    for k, c in COL.items():
        out[k] = num(c) if c in cols else np.nan
    tensor = [c for c in cols if TENSOR.fullmatch(c) and not TENSOR_SKIP.search(c)]
    out["tensor_flops"] = sum(num(c) for c in tensor) if tensor else 0.0
    fma = PLAIN_FMA if any(c in cols for c in PLAIN_FMA) else DERIVED_FMA
    out["cuda_flops"] = sum(w * num(c) for c, w in fma.items() if c in cols)
    out["flops"] = out.tensor_flops + out.cuda_flops
    if COL["rd"] in cols:
        out["bytes"] = out.rd + out.wr
    elif COL["dram_rate"] in cols:
        out["bytes"] = out.dram_rate * out.t
    else:
        raise SystemExit(f"{path}: no DRAM byte counters")
    out["klass"] = [next((k for k, p in CLASS_RULES if re.search(p, n.lower())), "Other") for n in out.name]
    return out[out.t > 0].reset_index(drop=True), tensor


def per_class(df):
    rows, T = [], df.t.sum()
    for k, g in df.groupby("klass"):
        f, b, t = g.flops.sum(), g.bytes.sum(), g.t.sum()
        w = g.t / t
        r = {"klass": k, "kernels": len(g), "time_share": t / T, "flops": f, "dram_bytes": b, "time_s": t,
             "ai_dram": f / b if b > 0 else np.nan, "tflops": f / t / 1e12}
        for lvl in ("l2", "l1"):
            r[f"ai_{lvl}"] = f / g[lvl].sum() if g[lvl].notna().all() and g[lvl].sum() > 0 else np.nan
        for s in ("sol_dram", "sol_l2", "sol_l1", "sol_tensor"):
            r[s] = float((g[s] * w).sum()) if g[s].notna().all() else np.nan
        rows.append(r)
    f, b, t = df.flops.sum(), df.bytes.sum(), T
    rows.append({"klass": "ALL", "kernels": len(df), "time_share": 1.0, "flops": f, "dram_bytes": b, "time_s": t,
                 "ai_dram": f / b if b > 0 else np.nan, "tflops": f / t / 1e12})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="*_raw.csv files or directories containing them")
    ap.add_argument("--out", default="roofline_out")
    ap.add_argument("--peak-tflops", type=float, default=0, help="measured BF16 GEMM ceiling (TFLOP/s)")
    ap.add_argument("--peak-tbs", type=float, default=0, help="measured DRAM bandwidth ceiling (TB/s)")
    ap.add_argument("--dense-peak-factor", type=float, default=0.5,
                    help="factor on the BF16 counter peak (0.5 on Hopper, whose counter peak is the 2:4-sparse rate)")
    a = ap.parse_args()
    files = sorted({f for x in a.inputs for f in (glob.glob(os.path.join(x, "*_raw.csv")) if os.path.isdir(x) else [x])})
    if not files:
        raise SystemExit("no input CSVs")
    os.makedirs(a.out, exist_ok=True)

    runs, report = [], []
    for f in files:
        stem = os.path.basename(f).removesuffix(".csv").removesuffix("_raw")
        m = next((m for t in TAGS if (m := t.fullmatch(stem))), None)
        label, phase = (m["run"], m["phase"]) if m else (stem, "step")
        df, tensor = load(f)
        s = per_class(df)
        s.insert(0, "phase", phase); s.insert(0, "run", label); s.insert(0, "file", os.path.basename(f))
        report.append(s)
        runs.append((label, phase, df, s))
        print(f"{os.path.basename(f)}: {len(df)} kernels, tensor counters {[c.split('src_')[1] for c in tensor]}")
    rep = pd.concat(report, ignore_index=True)
    rep.to_csv(os.path.join(a.out, "kernel_class_summary.csv"), index=False)

    labels = sorted({r[0] for r in runs})
    colors = dict(zip(labels, plt.get_cmap("tab10").colors))
    for phase in sorted({r[1] for r in runs}):
        sel = [r for r in runs if r[1] == phase]
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        sm_hz, dram_hz, dram_pk, bf16_pk = [], [], [], []
        for label, _, df, s in sel:
            w = df.t / df.t.sum()
            sm_hz.append((df.sm_hz * w).sum()); dram_hz.append((df.dram_hz * w).sum())
            dram_pk.append(df.dram_pk.median()); bf16_pk.append(df.bf16_pk.median())
            for _, r in s[s.klass != "ALL"].iterrows():
                if r.ai_dram > 0 and r.tflops > 0:
                    ax.scatter(r.ai_dram, r.tflops, s=40 + 900 * r.time_share, c=[colors[label]],
                               marker=CLASS_SHAPES[r.klass], alpha=0.8, edgecolors="black", linewidths=0.6, zorder=5)
        if a.peak_tflops and a.peak_tbs:
            bw, pk, src = a.peak_tbs, a.peak_tflops, "measured ceilings"
        else:
            bw = np.nanmedian(dram_pk) * np.nanmedian(dram_hz) / 1e12
            pk = np.nanmedian(bf16_pk) * a.dense_peak_factor * np.nanmedian(sm_hz) / 1e12
            src = f"counter peaks x measured clocks, BF16 peak x {a.dense_peak_factor:g}"
        if np.isfinite(bw) and np.isfinite(pk):
            ai = np.logspace(-2, 4, 400)
            ax.plot(ai, np.minimum(ai * bw, pk), color="#111111", lw=2.0, zorder=3)
            ax.set_ylim(1e-3, pk * 3)
            title = f"{phase}: roof {bw:.2f} TB/s, {pk:.0f} TFLOP/s ({src})"
        else:
            title = f"{phase}: no roof (pass --peak-tflops and --peak-tbs)"
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-2, 1e4)
        ax.set_xlabel("Arithmetic intensity (executed FLOP per DRAM byte)")
        ax.set_ylabel("Performance (executed TFLOP/s)")
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="major", alpha=0.25)
        ax.add_artist(ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=9, markerfacecolor=colors[l],
                                                markeredgecolor="black", label=l) for l in labels if any(r[0] == l for r in sel)],
                                loc="upper left", fontsize=8))
        ax.legend(handles=[Line2D([0], [0], marker=CLASS_SHAPES[c], ls="", ms=9, markerfacecolor="#999999",
                                  markeredgecolor="black", label=c) for c in CLASS_ORDER], loc="lower right", fontsize=8)
        fig.tight_layout()
        out = os.path.join(a.out, f"roofline_{phase}")
        fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=130)
        plt.close(fig)
        print(title, "->", out + ".pdf")
    cols = ["run", "phase", "klass", "kernels", "time_share", "ai_dram", "tflops", "ai_l2", "sol_dram", "sol_l2", "sol_l1", "sol_tensor"]
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(rep[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
