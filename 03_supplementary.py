#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

BACKLASH_CATEGORIES = [
    "challenges_refusal", "repeats_original_request",
    "clarification_request", "uncategorized",
]

BASE_INT = "disclaimer * framing + disclaimer * apology + framing * apology"

_LOG = []


def log(msg=""):
    print(msg, flush=True)
    _LOG.append(str(msg))


def section(t):
    log("\n" + "=" * 78)
    log(t)
    log("=" * 78)


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "+"
    return ""


def load(path):
    df = pd.read_csv(path, low_memory=False)
    ref = df["is_actual_refusal"].astype(str).str.strip().str.upper()
    d = df[ref == "TRUE"].copy().reset_index(drop=True)
    d["backlash"] = d["user_response_category"].isin(BACKLASH_CATEGORIES).astype(int)
    d["disclaimer"] = d["has_ai_identity_disclaimer"].astype(int)
    d["framing"] = d["has_normative_value_signaling"].astype(int)
    d["apology"] = d["has_apology_regret"].astype(int)
    if "english" not in d.columns:
        d["english"] = (d["language"] == "English").astype(float)
        d.loc[d["language"] == "unknown", "english"] = np.nan
    if "topic" not in d.columns:
        d["topic"] = d["prompt_category"].astype(str)
    if "model_grp" not in d.columns:
        vc = d["model"].value_counts()
        d["model_grp"] = d["model"].where(~d["model"].isin(vc[vc < 100].index),
                                          "other_small")
    return d


def cell_counts(d, focal, moderator):
    rows = []
    for mdl, g in d.groupby("model_grp"):
        row = {"model_grp": mdl, "n": len(g)}
        cells = []
        for f in (0, 1):
            for m in (0, 1):
                sub = g[(g[focal] == f) & (g[moderator] == m)]
                row[f"n_{focal[:3]}{f}_{moderator[:3]}{m}"] = len(sub)
                row[f"bl_{focal[:3]}{f}_{moderator[:3]}{m}"] = (
                    round(100 * sub["backlash"].mean(), 1) if len(sub) else np.nan)
                cells.append(len(sub))
        row["min_cell"] = min(cells)
        row["identifies"] = "yes" if min(cells) >= 20 else "no"
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="results_supplementary")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    section("0. SAMPLE")
    d = load(args.input)
    log(f"  confirmed refusals: {len(d):,}")
    log(f"  models (grouped): {d['model_grp'].nunique()}")
    has_tox = "toxicity_max" in d.columns and d["toxicity_max"].notna().any()
    if has_tox:
        d["tox_max_c"] = d["toxicity_max"].fillna(d["toxicity_max"].median())
        d["tox_max_c"] = d["tox_max_c"] - d["tox_max_c"].mean()
    log(f"  toxicity available: {has_tox}")

    section("S1. CAN EACH MODEL IDENTIFY THE INTERACTIONS?")

    log("\nDisclaimer x Normative framing, cell counts per LLM")
    cc_f = cell_counts(d, "disclaimer", "framing")
    log(cc_f.to_string(index=False))
    cc_f.to_csv(os.path.join(args.outdir, "cells_disclaimer_x_framing.csv"),
                index=False)

    log("\nDisclaimer x Apology, cell counts per LLM")
    cc_a = cell_counts(d, "disclaimer", "apology")
    log(cc_a.to_string(index=False))
    cc_a.to_csv(os.path.join(args.outdir, "cells_disclaimer_x_apology.csv"),
                index=False)

    id_f = set(cc_f.loc[cc_f["identifies"] == "yes", "model_grp"])
    id_a = set(cc_a.loc[cc_a["identifies"] == "yes", "model_grp"])
    log(f"\n  models with all four cells >= 20 for H2: {len(id_f)} of "
        f"{d['model_grp'].nunique()}  -> {sorted(id_f)}")
    log(f"  models with all four cells >= 20 for H3: {len(id_a)} of "
        f"{d['model_grp'].nunique()}  -> {sorted(id_a)}")

    n_in_f = d["model_grp"].isin(id_f).sum()
    n_in_a = d["model_grp"].isin(id_a).sum()
    log(f"\n  episodes in H2-identifying models: {n_in_f:,} "
        f"({100 * n_in_f / len(d):.1f}%)")
    log(f"  episodes in H3-identifying models: {n_in_a:,} "
        f"({100 * n_in_a / len(d):.1f}%)")

    section("S2. RESTRICTED SAMPLE (MODELS THAT CAN IDENTIFY H2)")
    sub_f = d[d["model_grp"].isin(id_f)].copy()
    log(f"  N = {len(sub_f):,}, models = {sub_f['model_grp'].nunique()}")
    r = smf.logit(f"backlash ~ {BASE_INT}", data=sub_f).fit(disp=0, maxiter=200)
    for t, h in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
        log(f"    {h}: B = {r.params[t]:+.3f}, SE = {r.bse[t]:.3f}, "
            f"p = {r.pvalues[t]:.4f} {stars(r.pvalues[t])}")

    M = "C(model_grp, Treatment('vicuna-13b'))"
    try:
        r2 = smf.logit(f"backlash ~ {BASE_INT} + {M}",
                       data=sub_f).fit(disp=0, maxiter=200)
        log("\n  same restricted sample, with model fixed effects:")
        for t, h in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
            log(f"    {h}: B = {r2.params[t]:+.3f}, SE = {r2.bse[t]:.3f}, "
                f"p = {r2.pvalues[t]:.4f} {stars(r2.pvalues[t])}")
    except Exception as e:
        log(f"  fixed-effects version failed: {e}")

    section("S3. RANDOM-INTERCEPT MODEL BY LLM")
    log("Fixed effects spend 19 df on nuisance parameters across unbalanced")
    log("clusters. A random intercept pools them and is the better-specified")
    log("model for 20 LLMs of very unequal size.")

    formula = f"backlash ~ {BASE_INT} + turn + english"
    if has_tox:
        formula += " + tox_max_c"
    dm = d.dropna(subset=["english"]).copy()

    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        vc = {"model_grp": "0 + C(model_grp)"}
        bm = BinomialBayesMixedGLM.from_formula(formula, vc, dm)
        rb = bm.fit_vb(verbose=False)
        log(f"\n  N = {len(dm):,}, random intercept over "
            f"{dm['model_grp'].nunique()} LLMs")
        rows = []
        for i, name in enumerate(rb.model.exog_names):
            mean = rb.fe_mean[i]
            sd = rb.fe_sd[i]
            z = mean / sd if sd > 0 else np.nan
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            rows.append({"term": name, "B": round(mean, 4),
                         "SD": round(sd, 4), "z": round(z, 3),
                         "p": round(p, 4), "sig": stars(p)})
        rb_tab = pd.DataFrame(rows)
        log(rb_tab.to_string(index=False))
        rb_tab.to_csv(os.path.join(args.outdir, "random_intercept_model.csv"),
                      index=False)
        log(f"\n  random-intercept SD (between-LLM): "
            f"{np.exp(rb.vcp_mean[0]):.4f}")
    except Exception as e:
        log(f"  random-intercept model failed: {e}")

    section("S4. POPULATION-AVERAGED GEE, EXCHANGEABLE BY LLM")
    try:
        gee = sm.GEE.from_formula(formula, groups="model_grp", data=dm,
                                  family=sm.families.Binomial(),
                                  cov_struct=sm.cov_struct.Exchangeable())
        rg = gee.fit()
        log(f"\n  N = {len(dm):,}, clusters = {dm['model_grp'].nunique()}")
        rows = []
        for t in ["disclaimer", "framing", "apology", "disclaimer:framing",
                  "disclaimer:apology", "framing:apology", "turn", "english",
                  "tox_max_c"]:
            if t in rg.params.index:
                rows.append({"term": t, "B": round(rg.params[t], 4),
                             "SE": round(rg.bse[t], 4),
                             "z": round(rg.tvalues[t], 3),
                             "p": round(rg.pvalues[t], 4),
                             "sig": stars(rg.pvalues[t])})
        gee_tab = pd.DataFrame(rows)
        log(gee_tab.to_string(index=False))
        gee_tab.to_csv(os.path.join(args.outdir, "gee_model.csv"), index=False)
    except Exception as e:
        log(f"  GEE failed: {e}")

    section("S5. CLUSTER BOOTSTRAP BY LLM")
    log(f"Resampling whole LLMs with replacement, {args.boot} replications.")
    log("This is the honest inference if you treat the 20 LLMs as the")
    log("population of interest rather than as fixed categories.")

    rng = np.random.RandomState(20251007)
    clusters = d["model_grp"].unique()
    groups = {c: d[d["model_grp"] == c] for c in clusters}
    boot = {"disclaimer:framing": [], "disclaimer:apology": [],
            "framing": [], "apology": []}

    ok = 0
    for b in range(args.boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        bs = pd.concat([groups[c] for c in pick], ignore_index=True)
        try:
            rb2 = smf.logit(f"backlash ~ {BASE_INT}", data=bs).fit(disp=0, maxiter=100)
            for k in boot:
                if k in rb2.params.index:
                    boot[k].append(rb2.params[k])
            ok += 1
        except Exception:
            continue
        if (b + 1) % 200 == 0:
            log(f"    {b + 1}/{args.boot} replications")

    log(f"\n  {ok}/{args.boot} replications converged")
    rows = []
    point = smf.logit(f"backlash ~ {BASE_INT}", data=d).fit(disp=0, maxiter=200)
    for k, v in boot.items():
        if not v:
            continue
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        p_boot = 2 * min((v <= 0).mean(), (v >= 0).mean())
        rows.append({"term": k, "B_point": round(point.params[k], 4),
                     "boot_mean": round(v.mean(), 4),
                     "boot_SE": round(v.std(ddof=1), 4),
                     "CI_low": round(lo, 4), "CI_high": round(hi, 4),
                     "p_boot": round(p_boot, 4),
                     "excludes_zero": "yes" if lo * hi > 0 else "no"})
    bt = pd.DataFrame(rows)
    log(bt.to_string(index=False))
    bt.to_csv(os.path.join(args.outdir, "cluster_bootstrap.csv"), index=False)

    section("S6. LEAVE-ONE-MODEL-OUT")
    log("Is either interaction driven by a single LLM?")
    rows = []
    for mdl in sorted(d["model_grp"].unique()):
        sub = d[d["model_grp"] != mdl]
        try:
            rr = smf.logit(f"backlash ~ {BASE_INT}", data=sub).fit(disp=0, maxiter=200)
            rows.append({
                "dropped": mdl, "N": len(sub),
                "H2_B": round(rr.params["disclaimer:framing"], 3),
                "H2_p": round(rr.pvalues["disclaimer:framing"], 4),
                "H3_B": round(rr.params["disclaimer:apology"], 3),
                "H3_p": round(rr.pvalues["disclaimer:apology"], 4)})
        except Exception:
            continue
    loo = pd.DataFrame(rows)
    log(loo.to_string(index=False))
    loo.to_csv(os.path.join(args.outdir, "leave_one_model_out.csv"), index=False)
    log(f"\n  H2 p < .05 after dropping any single LLM: "
        f"{(loo['H2_p'] < .05).sum()}/{len(loo)}")
    log(f"  H3 p < .05 after dropping any single LLM: "
        f"{(loo['H3_p'] < .05).sum()}/{len(loo)}")

    section("S7. LANGUAGE ROBUSTNESS")
    for lab, sub in [("English only", d[d["language"] == "English"]),
                     ("non-English only", d[(d["language"] != "English") &
                                            (d["language"] != "unknown")])]:
        if len(sub) < 300:
            log(f"  {lab}: N = {len(sub)} - too small")
            continue
        try:
            rr = smf.logit(f"backlash ~ {BASE_INT}", data=sub).fit(disp=0, maxiter=200)
            log(f"\n  {lab} (N = {len(sub):,}, "
                f"backlash {100 * sub['backlash'].mean():.1f}%):")
            for t, h in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
                log(f"    {h}: B = {rr.params[t]:+.3f}, SE = {rr.bse[t]:.3f}, "
                    f"p = {rr.pvalues[t]:.4f} {stars(rr.pvalues[t])}")
        except Exception as e:
            log(f"  {lab}: failed ({e})")

    section("S8. BACKLASH COMPONENTS")
    log("Does the pattern hold for the strict definition of backlash?")
    d["backlash_strict"] = d["user_response_category"].isin(
        ["challenges_refusal", "repeats_original_request"]).astype(int)
    log(f"  strict backlash: {int(d['backlash_strict'].sum()):,} "
        f"({100 * d['backlash_strict'].mean():.1f}%) vs "
        f"broad {int(d['backlash'].sum()):,} ({100 * d['backlash'].mean():.1f}%)")
    rr = smf.logit(f"backlash_strict ~ {BASE_INT}", data=d).fit(disp=0, maxiter=200)
    for t, h in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
        log(f"    {h}: B = {rr.params[t]:+.3f}, SE = {rr.bse[t]:.3f}, "
            f"p = {rr.pvalues[t]:.4f} {stars(rr.pvalues[t])}")

    log("\nChallenges refusal only (the single largest category):")
    d["bl_challenge"] = (d["user_response_category"] == "challenges_refusal").astype(int)
    rr = smf.logit(f"bl_challenge ~ {BASE_INT}", data=d).fit(disp=0, maxiter=200)
    for t, h in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
        log(f"    {h}: B = {rr.params[t]:+.3f}, SE = {rr.bse[t]:.3f}, "
            f"p = {rr.pvalues[t]:.4f} {stars(rr.pvalues[t])}")

    section("DONE")
    log(f"Written to {os.path.abspath(args.outdir)}")

    with open(os.path.join(args.outdir, "supplementary_log.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_LOG))


if __name__ == "__main__":
    main()
