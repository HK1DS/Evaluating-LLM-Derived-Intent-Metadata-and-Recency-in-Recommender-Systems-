#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User-activity analysis of OFF -> ON switches (11/24 plan, Jinwook's part) and the
base / meta-only / recency-only / both table (after the 12 new runs).

Inputs are the per-user files written by eval_slices.py
(<dir>/<spec>__s<seed>__<phase>.csv.gz with user_id, activity_train and metrics).

pair  — one OFF/ON comparison on the same backbone:
  1. join OFF and ON by user_id within each seed; delta = ON - OFF (default NDCG@10)
  2. average each user's delta over the seeds common to both (seeds are never resampled
     as separate users)
  3. five activity buckets by train interactions (activity_train): value quantile edges,
     right-closed, so users with the same activity always share a bucket and ties at an
     edge go to the lower bucket; edges and real bucket sizes are written out
  4. per bucket and overall: users, mean OFF/ON, mean delta (+ relative), user-level
     bootstrap 95% CI (fixed RNG seed), improved / tied / worse shares (|delta| <= tol is a
     tie), and the mean delta of every seed (direction check)

table — per spec: seed mean and std (ddof=0) of the result-JSON metrics, plus paired
  deltas against a reference spec with user CIs. Missing specs/seeds are shown as missing.

Usage (repository root)
  python scripts/analyze_activity_ablation.py pair --label "MovieLens MF" \
      --off_dir run_k30_TO/per_user_TO --off IKGR_kgoff --on IKGR_dyn_nointent \
      --out analysis/activity_TO/movielens_mf
  python scripts/analyze_activity_ablation.py table --label MovieLens --ref IKGR_kgoff \
      --spec IKGR_kgoff=run_k30_TO --spec IKGR_dyn_nointent=run_k30_TO \
      --spec IKGR_meta_only_nov22=run_k30_TO/ablation_nov22 \
      --spec IKGR_recency_only_nov22=run_k30_TO/ablation_nov22 --out analysis/activity_TO/movielens_table
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

N_BOOT = 10_000
RNG_SEED = 20261124
TIE_TOL = 1e-9
SEEDS = ("2020", "2021", "2022")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def per_user_path(d, spec, seed, phase):
    return os.path.join(d, f"{spec}__s{seed}__{phase}.csv.gz")


def load(d, spec, seed, phase, metric):
    p = per_user_path(d, spec, seed, phase)
    if not os.path.exists(p):
        return None, None
    df = pd.read_csv(p, usecols=["user_id", "activity_train", metric], dtype={"user_id": str})
    if df["user_id"].duplicated().any():
        raise SystemExit(f"duplicate user_id in {p}")
    if not np.isfinite(df[metric].to_numpy(dtype=float)).all():
        raise SystemExit(f"NaN/Inf {metric} in {p}")
    return df, {"file": p, "sha256_16": _sha(p), "users": len(df)}


def boot_ci(x, rng, n_boot=N_BOOT):
    """95% percentile CI of the mean, resampling users (chunked multinomial weights)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    means = np.empty(n_boot)
    done = 0
    chunk = max(1, min(n_boot, 20_000_000 // max(n, 1)))
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(k, n), dtype=np.int32)
        means[done:done + k] = x[idx].mean(axis=1)
        done += k
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def activity_buckets(act, q=5):
    """Right-closed value-quantile edges: equal activity -> same bucket, edge ties go down."""
    edges = np.quantile(act, np.linspace(0, 1, q + 1))
    inner = edges[1:-1]
    if len(np.unique(inner)) < len(inner):
        raise SystemExit(f"activity quantile edges collapse: {inner}")
    b = np.searchsorted(inner, act, side="left")       # act <= edge_k -> bucket k
    return b, edges


def pair(args):
    rng = np.random.default_rng(RNG_SEED)
    on_dir = args.on_dir or args.off_dir
    merged, inputs, used = [], [], []
    for s in args.seeds:
        off, i_off = load(args.off_dir, args.off, s, args.phase, args.metric)
        on, i_on = load(on_dir, args.on, s, args.phase, args.metric)
        if off is None or on is None:
            print(f"[skip seed {s}] missing {'OFF' if off is None else 'ON'} file")
            continue
        if set(off.user_id) != set(on.user_id):
            raise SystemExit(f"seed {s}: OFF and ON have different user sets")
        m = off.merge(on, on="user_id", suffixes=("_off", "_on"))
        if (m.activity_train_off != m.activity_train_on).any():
            raise SystemExit(f"seed {s}: activity_train differs between OFF and ON")
        m = m.rename(columns={"activity_train_off": "activity"}).drop(columns="activity_train_on")
        m["seed"] = s
        merged.append(m)
        inputs += [i_off, i_on]
        used.append(s)
    if len(used) < 2:
        raise SystemExit(f"need >= 2 common seeds, have {used}")
    allm = pd.concat(merged)
    acts = allm.groupby("user_id")["activity"].nunique()
    if (acts > 1).any():
        raise SystemExit("activity_train differs across seeds for some users")
    met = args.metric
    allm["delta"] = allm[f"{met}_on"] - allm[f"{met}_off"]
    per_seed = allm.pivot(index="user_id", columns="seed", values="delta")
    if per_seed.isna().any().any():
        raise SystemExit("users missing in some seed")
    u = allm.groupby("user_id").agg(activity=("activity", "first"), off=(f"{met}_off", "mean"),
                                    on=(f"{met}_on", "mean"), delta=("delta", "mean"))
    u = u.join(per_seed.add_prefix("delta_s"))
    u["bucket"], edges = activity_buckets(u["activity"].to_numpy())

    rows = []
    groups = [("all", u)] + [(f"Q{k + 1}", u[u.bucket == k]) for k in range(5)]
    for name, g in groups:
        lo, hi = boot_ci(g["delta"].to_numpy(), rng)
        d = g["delta"].to_numpy()
        row = {"bucket": name, "users": len(g),
               "activity_min": int(g.activity.min()), "activity_max": int(g.activity.max()),
               "mean_off": g.off.mean(), "mean_on": g.on.mean(), "mean_delta": d.mean(),
               "rel_delta_pct": 100 * d.mean() / g.off.mean() if g.off.mean() else float("nan"),
               "ci_lo": lo, "ci_hi": hi,
               "improved": float((d > TIE_TOL).mean()), "tied": float((np.abs(d) <= TIE_TOL).mean()),
               "worse": float((d < -TIE_TOL).mean())}
        for s in used:
            row[f"delta_s{s}"] = g[f"delta_s{s}"].mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    meta = {"label": args.label, "off": args.off, "on": args.on, "metric": met, "phase": args.phase,
            "seeds": used, "users": int(len(u)), "activity_edges": [float(e) for e in edges],
            "bucket_rule": "value quantile edges (numpy linear), right-closed; equal activity -> same "
                           "bucket, ties at an edge go to the lower bucket",
            "bootstrap": {"n": N_BOOT, "rng_seed": RNG_SEED, "unit": "user (seed-averaged delta)"},
            "tie_tolerance": TIE_TOL, "inputs": inputs}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out + ".csv", index=False)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False, indent=1, default=float)
    md = pair_markdown(meta, table)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


def pair_markdown(meta, t):
    seeds = meta["seeds"]
    lines = [f"### {meta['label']}: `{meta['off']}` (OFF) → `{meta['on']}` (ON), {meta['metric']} {meta['phase']}", "",
             f"users {meta['users']:,} · seeds {', '.join(seeds)} · activity edges "
             f"{', '.join(str(int(e)) for e in meta['activity_edges'])}", "",
             "| bucket | users | train inter. | OFF | ON | Δ (ON−OFF) | Δ % | 95% CI | improved / tie / worse | "
             + " | ".join(f"Δ s{s}" for s in seeds) + " |",
             "|---|---:|---|---:|---:|---:|---:|---|---|" + "---:|" * len(seeds)]
    for r in t.to_dict("records"):
        lines.append(f"| {r['bucket']} | {r['users']:,} | {r['activity_min']}–{r['activity_max']} | {r['mean_off']:.4f} | "
                     f"{r['mean_on']:.4f} | {r['mean_delta']:+.4f} | {r['rel_delta_pct']:+.1f}% | "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['improved']:.0%} / {r['tied']:.0%} / {r['worse']:.0%} | "
                     + " | ".join(f"{r[f'delta_s{s}']:+.4f}" for s in seeds) + " |")
    return "\n".join(lines) + "\n"


def table(args):
    rng = np.random.default_rng(RNG_SEED)
    specs = []
    for item in args.spec:
        name, _, d = item.partition("=")
        specs.append((name, d))
    agg_keys = [("NDCG@10", lambda v: v["overall"]["ndcg@10"]), ("Recall@10", lambda v: v["overall"]["recall@10"]),
                ("tail Recall@10", lambda v: v["long_tail"]["recall@10"]),
                ("tail Recall@30", lambda v: v["long_tail"].get("recall@30")),
                ("Coverage@10", lambda v: v.get("coverage@10")), ("Novelty", lambda v: v.get("novelty_bits"))]
    rows, per_user = [], {}
    for name, d in specs:
        res = os.path.join(d, f"slice_eval_{args.split}_result.json")
        seeds = {}
        if os.path.exists(res):
            seeds = json.load(open(res, encoding="utf-8")).get(name, {}).get("seeds", {})
        row = {"spec": name, "source": d, "seeds": ",".join(sorted(seeds)) or "—"}
        for key, get in agg_keys:
            vals = [get(seeds[s]) for s in sorted(seeds) if get(seeds[s]) is not None]
            row[key] = (float(np.mean(vals)), float(np.std(vals, ddof=0))) if vals else None
        rows.append(row)
        frames = []
        for s in args.seeds:
            df, _ = load(os.path.join(d, f"per_user_{args.split}"), name, s, "test", args.metric)
            if df is not None:
                frames.append(df.assign(seed=s))
        if frames:
            per_user[name] = pd.concat(frames)
    lines = [f"### {args.label}: 2×2 ({args.metric} test, seed mean ± std ddof=0)", "",
             "| spec | seeds | " + " | ".join(k for k, _ in agg_keys) + " |", "|---|---|" + "---:|" * len(agg_keys)]
    for r in rows:
        cells = [f"{r[k][0]:.4f} ± {r[k][1]:.4f}" if r[k] else "—" for k, _ in agg_keys]
        lines.append(f"| `{r['spec']}` | {r['seeds']} | " + " | ".join(cells) + " |")
    ref = per_user.get(args.ref)
    pairs = []
    if ref is not None:
        lines += ["", f"paired vs `{args.ref}` (users, common seeds averaged per user, bootstrap {N_BOOT:,}):", "",
                  "| spec | common seeds | users | Δ | Δ % | 95% CI |", "|---|---|---:|---:|---:|---|"]
        for name, _ in specs:
            if name == args.ref or name not in per_user:
                continue
            a, b = per_user[name], ref
            common = sorted(set(a.seed) & set(b.seed))
            if len(common) < 2:
                lines.append(f"| `{name}` | {','.join(common) or '—'} | — | — | — | needs ≥ 2 common seeds |")
                continue
            m = a[a.seed.isin(common)].merge(b[b.seed.isin(common)], on=["user_id", "seed"], suffixes=("_a", "_b"))
            m["delta"] = m[f"{args.metric}_a"] - m[f"{args.metric}_b"]
            u = m.groupby("user_id").agg(delta=("delta", "mean"), base=(f"{args.metric}_b", "mean"))
            lo, hi = boot_ci(u["delta"].to_numpy(), rng)
            d = u["delta"].mean()
            pairs.append({"spec": name, "ref": args.ref, "seeds": common, "users": len(u), "delta": d,
                          "rel_pct": 100 * d / u.base.mean(), "ci": [lo, hi]})
            lines.append(f"| `{name}` | {','.join(common)} | {len(u):,} | {d:+.4f} | {100 * d / u.base.mean():+.1f}% | "
                         f"[{lo:+.4f}, {hi:+.4f}] |")
    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "pairs": pairs, "bootstrap": {"n": N_BOOT, "rng_seed": RNG_SEED}}, f,
                  ensure_ascii=False, indent=1, default=float)
    print(md)


def factorial(args):
    """2x2 decomposition on users: meta and recency main effects, interaction, overall and per activity bucket.

    For each user the four cells are seed-averaged NDCG first (common seeds of all four cells). Effects:
      meta     = ((meta_only - base) + (both - recency_only)) / 2
      recency  = ((recency_only - base) + (both - meta_only)) / 2
      interaction = both - meta_only - recency_only + base
    plus the simple effects (each switch alone vs base, and each switch added to the other)."""
    rng = np.random.default_rng(RNG_SEED)
    cells = {}
    for key in ("base", "meta", "rec", "both"):
        name, _, d = getattr(args, key).partition("=")
        frames = [df.assign(seed=s) for s in args.seeds
                  for df, _ in [load(os.path.join(d, f"per_user_{args.split}"), name, s, args.phase, args.metric)]
                  if df is not None]
        if not frames:
            raise SystemExit(f"no per-user files for {name} in {d}")
        cells[key] = (name, pd.concat(frames))
    common = sorted(set.intersection(*(set(df.seed) for _, df in cells.values())))
    if len(common) < 2:
        raise SystemExit(f"need >= 2 seeds common to all four cells, have {common}")
    u = None
    for key, (name, df) in cells.items():
        g = df[df.seed.isin(common)].groupby("user_id").agg(**{key: (args.metric, "mean"), f"act_{key}": ("activity_train", "first")})
        u = g if u is None else u.join(g, how="inner")
    acts = u[[c for c in u.columns if c.startswith("act_")]]
    if (acts.nunique(axis=1) > 1).any():
        raise SystemExit("activity_train differs between cells")
    u["activity"] = acts.iloc[:, 0]
    u["bucket"], edges = activity_buckets(u["activity"].to_numpy())
    effects = {
        "meta alone (meta − base)": u.meta - u.base,
        "recency alone (rec − base)": u.rec - u.base,
        "both (both − base)": u.both - u.base,
        "meta added to recency (both − rec)": u.both - u.rec,
        "recency added to meta (both − meta)": u.both - u.meta,
        "meta main effect": ((u.meta - u.base) + (u.both - u.rec)) / 2,
        "recency main effect": ((u.rec - u.base) + (u.both - u.meta)) / 2,
        "interaction": u.both - u.meta - u.rec + u.base,
    }
    rows = []
    for ename, e in effects.items():
        for bname, mask in [("all", np.ones(len(u), bool))] + [(f"Q{k + 1}", (u.bucket == k).to_numpy()) for k in range(5)]:
            x = e.to_numpy()[mask]
            lo, hi = boot_ci(x, rng)
            rows.append({"effect": ename, "bucket": bname, "users": int(mask.sum()), "delta": float(x.mean()),
                         "rel_pct": float(100 * x.mean() / u.base.to_numpy()[mask].mean()), "ci_lo": lo, "ci_hi": hi})
    t = pd.DataFrame(rows)
    names = {k: v[0] for k, v in cells.items()}
    lines = [f"### {args.label}: 2×2 분해 ({args.metric} {args.phase}, 사용자 단위, seed {', '.join(common)} 평균)", "",
             f"base `{names['base']}` · meta `{names['meta']}` · recency `{names['rec']}` · both `{names['both']}` · "
             f"users {len(u):,} · activity edges {', '.join(str(int(e)) for e in edges)}", "",
             "| cell | " + " | ".join(["all"] + [f"Q{k}" for k in range(1, 6)]) + " |", "|---|" + "---:|" * 6]
    for key, label in (("base", "base"), ("meta", "meta only"), ("rec", "recency only"), ("both", "both")):
        vals = [u[key].mean()] + [u[key][u.bucket == k].mean() for k in range(5)]
        lines.append(f"| {label} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines += ["", "| effect | " + " | ".join(["all"] + [f"Q{k}" for k in range(1, 6)]) + " |", "|---|" + "---|" * 6]
    for ename in effects:
        sub = t[t.effect == ename].set_index("bucket")
        cells_txt = []
        for b in ["all"] + [f"Q{k}" for k in range(1, 6)]:
            r = sub.loc[b]
            sig = "**" if (r.ci_lo > 0 or r.ci_hi < 0) else ""
            cells_txt.append(f"{sig}{r.rel_pct:+.1f}%{sig} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]")
        lines.append(f"| {ename} | " + " | ".join(cells_txt) + " |")
    lines += ["", "굵게 = 95% CI가 0을 포함하지 않음. % = Δ / base 평균."]
    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t.to_csv(args.out + ".csv", index=False)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"cells": names, "seeds": common, "users": int(len(u)), "activity_edges": [float(e) for e in edges],
                   "bootstrap": {"n": N_BOOT, "rng_seed": RNG_SEED}, "rows": rows}, f, ensure_ascii=False, indent=1)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    fa = sub.add_parser("factorial")
    fa.add_argument("--label", required=True)
    fa.add_argument("--split", default="TO")
    for key in ("base", "meta", "rec", "both"):
        fa.add_argument(f"--{key}", required=True, help="NAME=RUN_DIR")
    fa.add_argument("--seeds", nargs="+", default=list(SEEDS))
    fa.add_argument("--phase", default="test", choices=["test", "valid"])
    fa.add_argument("--metric", default="ndcg@10")
    fa.add_argument("--out", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--label", required=True)
    p.add_argument("--off_dir", required=True)
    p.add_argument("--on_dir", default=None, help="default: same as --off_dir")
    p.add_argument("--off", required=True)
    p.add_argument("--on", required=True)
    p.add_argument("--seeds", nargs="+", default=list(SEEDS))
    p.add_argument("--phase", default="test", choices=["test", "valid"])
    p.add_argument("--metric", default="ndcg@10")
    p.add_argument("--out", required=True, help="output prefix (.csv/.json/.md)")
    t = sub.add_parser("table")
    t.add_argument("--label", required=True)
    t.add_argument("--split", default="TO")
    t.add_argument("--spec", action="append", required=True, help="NAME=RUN_DIR (result JSON + per_user_<SPLIT>/)")
    t.add_argument("--ref", required=True)
    t.add_argument("--seeds", nargs="+", default=list(SEEDS))
    t.add_argument("--metric", default="ndcg@10")
    t.add_argument("--out", required=True)
    args = ap.parse_args()
    {"pair": pair, "table": table, "factorial": factorial}[args.cmd](args)


if __name__ == "__main__":
    main()
