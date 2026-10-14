"""Shared helpers for the step-1 data pull (read-only on results/)."""
import glob, json, os
from collections import Counter
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "contexts", "step1")
CANONICAL_SEEDS = {331, 958, 14577, 43611, 85062}
IGNORED_EXP = {"seed", "output_dir", "algo_name", "env_id"}


def _flat(d, prefix=""):
    return {prefix + k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in d.items()}


def load_all_meta():
    recs = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*", "*", "seed_*", "*", "metadata.json"))):
        rd = os.path.dirname(p)
        with open(p) as f:
            m = json.load(f)
        m["_dir"] = rd
        m["_rel"] = os.path.relpath(rd, RESULTS).replace("\\", "/")
        recs.append(m)
    return recs


def _label(flat, modal, keys):
    diffs = sorted(f"{k.removeprefix('exp.')}={flat[k]}" for k in keys if k in flat and flat[k] != modal[k])
    return ", ".join(diffs) if diffs else "baseline"


def canonical_runs():
    """DataFrame of the canonical runs (+ meta dict in column 'meta').
    Config label = difference of the run's config from the modal config of
    its (algo, env) group among canonical runs (each sweep varies fields
    off the baseline; baseline is modal)."""
    metas = [m for m in load_all_meta() if m["seed"] in CANONICAL_SEEDS]
    flats = []
    for m in metas:
        f = {**_flat(m.get("algo_config", {})),
             **{k: v for k, v in _flat(m.get("experiment_config", {}), "exp.").items() if k[4:] not in IGNORED_EXP}}
        flats.append(f)
    modal = {}
    for (a, e) in {(m["algo_name"], m["env_id"]) for m in metas}:
        idx = [i for i, m in enumerate(metas) if (m["algo_name"], m["env_id"]) == (a, e)]
        keys = set().union(*(flats[i].keys() for i in idx))
        modal[(a, e)] = ({k: Counter(flats[i][k] for i in idx if k in flats[i]).most_common(1)[0][0] for k in keys}, keys)
    rows = []
    for m, f in zip(metas, flats):
        mo, keys = modal[(m["algo_name"], m["env_id"])]
        rows.append(dict(algo=m["algo_name"], env=m["env_id"], seed=m["seed"],
                         config=_label(f, mo, keys), run_dir=m["_rel"], start_time_utc=m["start_time_utc"],
                         end_time_utc=m.get("end_time_utc"), git_commit=m.get("git_commit"), meta=m))
    df = pd.DataFrame(rows).sort_values("start_time_utc").reset_index(drop=True)
    return df


def read_base_csv(run_dir_rel):
    p = glob.glob(os.path.join(RESULTS, run_dir_rel, "emissions_base_*.csv"))
    return pd.read_csv(p[0]) if p else None


if __name__ == "__main__":
    df = canonical_runs()
    out = df.drop(columns=["meta"])
    out.to_csv(os.path.join(OUT, "04_canonical_run_list.csv"), index=False)
    print(len(df))
    print(df.groupby(["algo", "env"]).size())
    print(df.groupby(["algo", "env", "config"]).size().to_string())
    print(df.groupby(["algo", "env"]).apply(lambda g: g.groupby("seed").size().unique().tolist()))
