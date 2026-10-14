"""
Row-wise verification of the baseline tables of Chapter 4 of the thesis against the values recomputed from the raw run files
(the generator of Appendix B is flop_analysis/appendix_tables.py, which also does a looser look-up of all numbers of 50 tables).

For each table tab:<algo>-base-{energy,jpf,time,comp} of thesis-main.tex (or its backup) the cells of every row are compared with the recomputed
value of the same (environment, segment, quantity).  A cell agrees if the recomputed value, rounded to the number of significant digits
printed, equals the printed number.  Prints one line per table and writes appendix_data/B/B_rowwise_verification.csv.

Usage: python flop_analysis/appendix_tables_verify.py [--thesis path]
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appendix_tables as a  # noqa: E402

BS = chr(92)
NUM = re.compile(re.escape(BS) + r"num\{([^{}]*)\}")
SEGKEY = {"rollout": "rollout", "buffer_sample": "buffer_sample", "critic_update": "critic_update", "actor_update": "actor_update",
          "target_update": "target_update", "gradient_updates": "gradient_updates", "dynamics_model_update": "dynamics_model_update",
          "synthetic_rollout_generation": "synthetic_rollout_generation", "world_model_pretrain": "world_model_pretrain"}


def cells_of(row: str):
    row = row.split(BS + BS)[0]
    return [c.strip() for c in row.split("&")]


def toks(cell: str):
    out = []
    for m in NUM.finditer(cell):
        s = m.group(1).replace("{,}", "")
        out.append(s.split("+-"))
    return out[0] if out else None            # [mean] or [mean, sd]


def seg_of(cell: str):
    c = cell.replace(BS + "_", "_")
    m = re.search(re.escape(BS) + r"texttt\{([^}]*)\}", c)
    if m:
        return SEGKEY.get(m.group(1))
    if c.strip().lower() == "total":
        return "TOTAL"
    return None


def agree(printed: str, mine: float):
    n = a.sigdigits(printed)[-1] if len(a.sigdigits(printed)) == 1 else a.sigdigits(printed)[0]
    ns = a.sigdigits(printed)
    ok = any(math.isclose(float(f"{abs(mine):.{k}g}"), abs(float(printed)), rel_tol=1e-9) for k in ns)
    dev = abs(abs(float(printed)) - abs(mine)) / abs(mine) if mine else abs(float(printed))
    return ok, dev


def block_rows(tex: str, label: str):
    blk = a.strip_caption(a.table_block(tex, label))
    return [l for l in blk.split("\n") if "&" in l and "multicolumn" not in l]


def check(tex, A, runs):
    res = []

    def add(label, key, printed, mine, sd=False):
        ok, dev = agree(printed, mine)
        res.append(dict(table=label, cell=key, printed=printed, recomputed=mine, agrees=ok, rel_dev=dev))

    for algo in a.ALGOS:
        # ---------------- energy
        lab = f"tab:{algo}-base-energy"
        if ("\\label{" + lab + "}") in tex:
            for row in block_rows(tex, lab):
                c = cells_of(row)
                seg = seg_of(c[0])
                if seg is None or len(c) < 8:
                    continue
                for ei, env in enumerate(a.ENVS):
                    d = A[(algo, env, "baseline")][seg]
                    cs = c[2 + 3 * ei: 5 + 3 * ei]
                    t = toks(cs[0])
                    add(lab, f"{env}/{seg}/E mean", t[0], float(np.mean(d["E"])))
                    add(lab, f"{env}/{seg}/E sd", t[1], float(np.std(d["E"], ddof=1)))
                    t = toks(cs[1])
                    if t and len(t) == 2:
                        add(lab, f"{env}/{seg}/share total mean", t[0], float(np.mean(d["share_total"])))
                        add(lab, f"{env}/{seg}/share total sd", t[1], float(np.std(d["share_total"], ddof=1)))
                    t = toks(cs[2])
                    if t and len(t) == 2 and d["share_gu"] is not None:
                        add(lab, f"{env}/{seg}/share GU mean", t[0], float(np.mean(d["share_gu"])))
                        add(lab, f"{env}/{seg}/share GU sd", t[1], float(np.std(d["share_gu"], ddof=1)))
        # ---------------- J/FLOP
        lab = f"tab:{algo}-base-jpf"
        if ("\\label{" + lab + "}") in tex:
            for row in block_rows(tex, lab):
                c = cells_of(row)
                seg = seg_of(c[0])
                if seg is None or len(c) < 5:
                    continue
                for ei, env in enumerate(a.ENVS):
                    d = A[(algo, env, "baseline")][seg]
                    fcell = re.sub(r"\s*ops", "", c[1 + 2 * ei])
                    t = toks(fcell)
                    if t:
                        add(lab, f"{env}/{seg}/FLOPs", t[0], float(np.mean(d["F"])))
                    t = toks(c[2 + 2 * ei])
                    if t and len(t) == 2:
                        f = 1e9 if seg == "target_update" else 1e12
                        add(lab, f"{env}/{seg}/J/FLOP mean", t[0], float(np.mean(d["E"]) / np.mean(d["F"]) * f))
                        add(lab, f"{env}/{seg}/J/FLOP sd", t[1], float(np.std(d["jpf_runs"] * f, ddof=1)))
        # ---------------- time, power, throughput
        lab = f"tab:{algo}-base-time"
        if ("\\label{" + lab + "}") in tex:
            for row in block_rows(tex, lab):
                c = cells_of(row)
                seg = seg_of(c[0])
                if seg is None or len(c) < 7:
                    continue
                for ei, env in enumerate(a.ENVS):
                    d = A[(algo, env, "baseline")][seg]
                    for k, (name, arr) in enumerate((("T", d["T"]), ("P", d["P"]), ("GFLOP/s", d["gflops"]))):
                        t = toks(c[1 + 3 * ei + k])
                        if t and len(t) == 2:
                            add(lab, f"{env}/{seg}/{name} mean", t[0], float(np.mean(arr)))
                            add(lab, f"{env}/{seg}/{name} sd", t[1], float(np.std(arr, ddof=1)))
        # ---------------- composition
        lab = f"tab:{algo}-base-comp"
        if ("\\label{" + lab + "}") in tex:
            env = None
            for row in block_rows(tex, lab):
                c = cells_of(row)
                if len(c) < 6:
                    continue
                if c[0]:
                    env = c[0]
                task = SEGKEY.get(c[1].replace(BS + "_", "_").replace(BS + "texttt{", "").replace("}", ""))
                if task is None or env not in a.ENVS:
                    continue
                cpu, gpu, ram, idle, _ = a.comp_arrays(runs, algo, env, "baseline", task)
                for name, arr, cell in (("CPU", cpu, c[2]), ("GPU", gpu, c[3]), ("RAM", ram, c[4]), ("idle floor", idle, c[5])):
                    t = toks(cell)
                    add(lab, f"{env}/{task}/{name} mean", t[0], float(np.mean(arr)))
                    add(lab, f"{env}/{task}/{name} sd", t[1], float(np.std(arr, ddof=1)))
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser()
    _t = os.path.join(os.path.dirname(os.path.dirname(a.REPO)), "Thesis Report", "thesis-main.tex")
    ap.add_argument("--thesis", default=(_t + ".bak-appendix") if os.path.exists(_t + ".bak-appendix") else _t)
    ap.add_argument("--out", default=os.path.join(a.CTX, "appendix", "appendix_data", "B", "B_rowwise_verification.csv"))
    args = ap.parse_args()
    tex = open(args.thesis, encoding="utf-8").read()
    runs = a.load_runs()
    A = a.agg(runs)
    df = check(tex, A, runs)
    df.to_csv(args.out, index=False)
    g = df.groupby("table").agg(cells=("agrees", "size"), agree=("agrees", "sum"), max_rel_dev=("rel_dev", "max"))
    print(g.to_string())
    print(f"total cells {len(df)}, agreeing {int(df.agrees.sum())}, max relative deviation {df.rel_dev.max():.3g}")
    bad = df[~df.agrees]
    if len(bad):
        print("disagreeing cells:")
        print(bad.to_string())


if __name__ == "__main__":
    main()
