"""Shared loader for section 7 scripts (read-only). Reuses flop_analysis/section_5_2_analysis.load() for run_dir<->config mapping."""
import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "flop_analysis"))
import pandas as pd, numpy as np
import section_5_2_analysis as s52
OUT = REPO / "contexts" / "section_7_data"
CANON = s52.CANON
def runs():
    r = s52.load()
    r["path"] = r.run_dir.map(lambda p: REPO / p)
    return r
def jload(p): return json.load(open(p))
