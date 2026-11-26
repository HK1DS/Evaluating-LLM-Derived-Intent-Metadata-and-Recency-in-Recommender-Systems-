#!/usr/bin/env python3
"""scripts/analyze_activity_ablation.py: seed-averaged user deltas, activity buckets with ties kept together,
refusal of mismatched inputs, and the 2x2 table with missing cells.

Run:  python -m pytest -q tests/test_activity_ablation.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "analyze_activity_ablation.py")


def _write(d, spec, seed, ndcg, act, phase="test"):
    n = len(ndcg)
    pd.DataFrame({"user_id": [f"u{i}" for i in range(n)], "activity_train": act, "ndcg@10": ndcg,
                  "recall@10": ndcg}).to_csv(os.path.join(d, f"{spec}__s{seed}__{phase}.csv.gz"),
                                             index=False, compression="gzip")


def _run(*args):
    p = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_pair_buckets_and_seed_average(tmp_path):
    act = np.repeat([1, 2, 3, 4, 5], 20)            # 100 users, 5 activity levels
    off = np.full(100, 0.10)
    for s, shift in (("2020", 0.0), ("2021", 0.02), ("2022", -0.02)):
        on = off + np.where(act <= 2, 0.05, -0.01) + shift        # low activity gains, high loses
        _write(tmp_path, "OFF", s, off, act)
        _write(tmp_path, "ON", s, on, act)
    rc, out = _run("pair", "--label", "t", "--off_dir", str(tmp_path), "--off", "OFF", "--on", "ON",
                   "--out", str(tmp_path / "res"))
    assert rc == 0, out
    res = json.load(open(tmp_path / "res.json"))
    rows = {r["bucket"]: r for r in res["rows"]}
    assert [rows[f"Q{k}"]["users"] for k in range(1, 6)] == [20] * 5          # equal activity -> one bucket
    assert abs(rows["Q1"]["mean_delta"] - 0.05) < 1e-9 and abs(rows["Q5"]["mean_delta"] + 0.01) < 1e-9
    assert rows["Q1"]["ci_lo"] > 0 > rows["Q5"]["ci_hi"] or rows["Q5"]["ci_lo"] == rows["Q5"]["ci_hi"]
    assert abs(rows["Q1"]["delta_s2021"] - 0.07) < 1e-9                       # per-seed direction kept
    assert rows["Q1"]["improved"] == 1.0 and rows["Q5"]["worse"] == 1.0


def test_pair_refuses_mismatched_activity(tmp_path):
    for s in ("2020", "2021"):
        _write(tmp_path, "OFF", s, np.full(10, 0.1), np.arange(10))
        _write(tmp_path, "ON", s, np.full(10, 0.2), np.arange(10) + (1 if s == "2021" else 0))
    rc, out = _run("pair", "--label", "t", "--off_dir", str(tmp_path), "--off", "OFF", "--on", "ON",
                   "--seeds", "2020", "2021", "--out", str(tmp_path / "res"))
    assert rc != 0 and "activity_train differs" in out


def test_table_marks_missing_cells(tmp_path):
    run = tmp_path / "run"
    (run / "per_user_TO").mkdir(parents=True)
    res = {}
    for spec, v in (("BASE", 0.1), ("BOTH", 0.12)):
        res[spec] = {"seeds": {s: {"overall": {"ndcg@10": v, "recall@10": v}, "long_tail": {"recall@10": 0.0},
                                   "coverage@10": 0.5, "novelty_bits": 10.0} for s in ("2020", "2021", "2022")}}
        for s in ("2020", "2021", "2022"):
            _write(run / "per_user_TO", spec, s, np.full(50, v), np.arange(50))
    json.dump(res, open(run / "slice_eval_TO_result.json", "w"))
    rc, out = _run("table", "--label", "t", "--ref", "BASE", "--spec", f"BASE={run}", "--spec", f"BOTH={run}",
                   "--spec", f"META={tmp_path / 'missing'}", "--out", str(tmp_path / "tab"))
    assert rc == 0, out
    md = (tmp_path / "tab.md").read_text(encoding="utf-8")
    assert "| `META` | — |" in md and "| `BOTH` | 2020,2021,2022 | 50 | +0.0200 | +20.0% |" in md


def test_factorial_effects(tmp_path):
    run = tmp_path / "run"
    (run / "per_user_TO").mkdir(parents=True)
    act = np.repeat([1, 2, 3, 4, 5], 10)
    for spec, v in (("BASE", 0.10), ("META", 0.10), ("REC", 0.15), ("BOTH", 0.14)):
        for s in ("2020", "2021", "2022"):
            _write(run / "per_user_TO", spec, s, np.full(50, v), act)
    rc, out = _run("factorial", "--label", "t", "--base", f"BASE={run}", "--meta", f"META={run}",
                   "--rec", f"REC={run}", "--both", f"BOTH={run}", "--out", str(tmp_path / "fac"))
    assert rc == 0, out
    rows = {(r["effect"], r["bucket"]): r for r in json.load(open(tmp_path / "fac.json"))["rows"]}
    assert abs(rows[("meta main effect", "all")]["delta"] + 0.005) < 1e-12
    assert abs(rows[("recency main effect", "all")]["delta"] - 0.045) < 1e-12
    assert abs(rows[("interaction", "Q3")]["delta"] + 0.01) < 1e-12
    assert abs(rows[("meta added to recency (both − rec)", "all")]["rel_pct"] + 10.0) < 1e-9
