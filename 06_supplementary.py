#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)

BASE = "/content/drive/MyDrive/Wasko_JPIM"
DEFAULT_IN = f"{BASE}/data/Wasko_Final_Dataset_ANALYSIS.csv"
DEFAULT_OUT = f"{BASE}/output/results_supplementary"

BACKLASH = ["challenges_refusal", "repeats_original_request", "clarification_request"]
FOCAL = "disclaimer * framing + disclaimer * apology + framing * apology"
SEED = 20251007

_LOG = []


def log(m=""):
    print(m, flush=True)
    _LOG.append(str(m))


def section(t):
    log("\n" + "=" * 78)
    log(t)
    log("=" * 78)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "†" if p < .10 else ""


def load(path):
    raw = pd.read_csv(path, low_memory=False)
    d = raw[raw["is_actual_refusal"].astype(str).str.strip().str.upper() == "TRUE"].copy()
    d = d.reset_index(drop=True)
    d["backlash"] = d["user_response_category"].isin(BACKLASH).astype(int)
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
        d["model_grp"] = d["model"].where(~d["model"].isin(vc[vc < 100].index), "other_small")
    return d


def cells(d, focal, moderator, threshold=20):
    rows = []
    for mdl, g in d.groupby("model_grp"):
        row = {"model_grp": mdl, "n": len(g)}
        counts = []
        for x in (0, 1):
            for m in (0, 1):
                sub = g[(g[focal] == x) & (g[moderator] == m)]
                row[f"n_{focal[:3]}{x}_{moderator[:3]}{m}"] = len(sub)
                row[f"bl_{focal[:3]}{x}_{moderator[:3]}{m}"] = (
                    round(100 * sub["backlash"].mean(), 1) if len(sub) else np.nan)
                counts.append(len(sub))
        row["min_cell"] = min(counts)
        row["identifies"] = "yes" if min(counts) >= threshold else "no"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def report(res, label):
    log(f"\n  {label}")
    for term, hyp in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
        if term in res.params.index:
            log(f"    {hyp}: B = {res.params[term]:+.3f}, SE = {res.bse[term]:.3f}, "
                f"p = {res.pvalues[term]:.4f} {stars(res.pvalues[term])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    section("0. SAMPLE")
    d = load(args.input)
    log(f"  confirmed refusals: {len(d):,}")
    log(f"  responding systems (grouped): {d['model_grp'].nunique()}")

    has_tox = "toxicity_max" in d.columns and d["toxicity_max"].notna().any()
    has_sev = "request_severity" in d.columns and d["request_severity"].notna().any()
    if has_tox:
        d["tox"] = d["toxicity_max"].fillna(d["toxicity_max"].median())
        d["tox_c"] = d["tox"] - d["tox"].mean()
    if has_sev:
        d["sev"] = d["request_severity"].fillna(d["request_severity"].median())
        d["sev_c"] = d["sev"] - d["sev"].mean()
    log(f"  toxicity: {has_tox}   severity: {has_sev}")

    dc = d.dropna(subset=["english"]).copy()
    top_topic = d["topic"].value_counts().idxmax()
    T = f"C(topic, Treatment('{top_topic}'))"
    M = "C(model_grp, Treatment('vicuna-13b'))"

    section("S1. CAN EACH SYSTEM IDENTIFY THE INTERACTIONS?")
    cf = cells(d, "disclaimer", "framing")
    log("\nDisclaimer x normative framing, cell counts by system")
    log(cf.to_string(index=False))
    cf.to_csv(os.path.join(args.outdir, "cells_disclaimer_x_framing.csv"), index=False)

    ca = cells(d, "disclaimer", "apology")
    log("\nDisclaimer x apology, cell counts by system")
    log(ca.to_string(index=False))
    ca.to_csv(os.path.join(args.outdir, "cells_disclaimer_x_apology.csv"), index=False)

    id_f = set(cf.loc[cf["identifies"] == "yes", "model_grp"])
    id_a = set(ca.loc[ca["identifies"] == "yes", "model_grp"])
    n_f = int(d["model_grp"].isin(id_f).sum())
    n_a = int(d["model_grp"].isin(id_a).sum())
    log(f"\n  systems with all four cells >= 20 for H2: {len(id_f)} of "
        f"{d['model_grp'].nunique()}  ({n_f:,} episodes, {100 * n_f / len(d):.1f}%)")
    log(f"  systems with all four cells >= 20 for H3: {len(id_a)} of "
        f"{d['model_grp'].nunique()}  ({n_a:,} episodes, {100 * n_a / len(d):.1f}%)")

    section("S2. RESTRICTED SAMPLES")
    sub_f = d[d["model_grp"].isin(id_f)]
    log(f"  systems that can identify H2: N = {len(sub_f):,}")
    report(smf.logit(f"backlash ~ {FOCAL}", data=sub_f).fit(disp=0, maxiter=200),
           "pooled")
    try:
        report(smf.logit(f"backlash ~ {FOCAL} + {M}", data=sub_f).fit(disp=0, maxiter=200),
               "with system fixed effects")
    except Exception as e:
        log(f"    fixed-effects version failed: {e}")

    sub_a = d[d["model_grp"].isin(id_a)]
    log(f"\n  systems that can identify H3: N = {len(sub_a):,}")
    report(smf.logit(f"backlash ~ {FOCAL}", data=sub_a).fit(disp=0, maxiter=200),
           "pooled")

    section("S3. POPULATION-AVERAGED GEE, EXCHANGEABLE BY SYSTEM")
    f_gee = f"backlash ~ {FOCAL} + turn + english"
    if has_tox:
        f_gee += " + tox_c"
    if has_sev:
        f_gee += " + sev_c"
    try:
        gee = sm.GEE.from_formula(f_gee, groups="model_grp", data=dc,
                                  family=sm.families.Binomial(),
                                  cov_struct=sm.cov_struct.Exchangeable()).fit()
        log(f"  N = {int(gee.nobs):,}, clusters = {dc['model_grp'].nunique()}")
        rows = []
        for t in ["disclaimer", "framing", "apology", "disclaimer:framing",
                  "disclaimer:apology", "framing:apology", "turn", "english",
                  "tox_c", "sev_c"]:
            if t in gee.params.index:
                log(f"    {t:20s} B = {gee.params[t]:+.4f}  SE = {gee.bse[t]:.4f}  "
                    f"p = {gee.pvalues[t]:.4f} {stars(gee.pvalues[t])}")
                rows.append({"term": t, "B": round(gee.params[t], 4),
                             "SE": round(gee.bse[t], 4),
                             "p": round(gee.pvalues[t], 4),
                             "sig": stars(gee.pvalues[t])})
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "gee_model.csv"), index=False)
    except Exception as e:
        log(f"  GEE failed: {e}")

    section("S4. CLUSTER BOOTSTRAP BY SYSTEM")
    log(f"Resampling whole systems with replacement, {args.boot} replications.")
    rng = np.random.RandomState(SEED)
    clusters = d["model_grp"].unique()
    groups = {c: d[d["model_grp"] == c] for c in clusters}
    boot = {"disclaimer:framing": [], "disclaimer:apology": [],
            "framing": [], "apology": []}
    ok = 0
    for b in range(args.boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        bs = pd.concat([groups[c] for c in pick], ignore_index=True)
        try:
            r = smf.logit(f"backlash ~ {FOCAL}", data=bs).fit(disp=0, maxiter=100)
            for k in boot:
                if k in r.params.index:
                    boot[k].append(r.params[k])
            ok += 1
        except Exception:
            continue
        if (b + 1) % 250 == 0:
            log(f"    {b + 1}/{args.boot}")
    log(f"\n  {ok}/{args.boot} replications converged")
    point = smf.logit(f"backlash ~ {FOCAL}", data=d).fit(disp=0, maxiter=200)
    rows = []
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

    section("S5. LEAVE-ONE-SYSTEM-OUT")
    rows = []
    for mdl in sorted(d["model_grp"].unique()):
        sub = d[d["model_grp"] != mdl]
        try:
            r = smf.logit(f"backlash ~ {FOCAL}", data=sub).fit(disp=0, maxiter=200)
            rows.append({"dropped": mdl, "N": len(sub),
                         "H2_B": round(r.params["disclaimer:framing"], 3),
                         "H2_p": round(r.pvalues["disclaimer:framing"], 4),
                         "H3_B": round(r.params["disclaimer:apology"], 3),
                         "H3_p": round(r.pvalues["disclaimer:apology"], 4)})
        except Exception:
            continue
    loo = pd.DataFrame(rows)
    log(loo.to_string(index=False))
    loo.to_csv(os.path.join(args.outdir, "leave_one_system_out.csv"), index=False)
    log(f"\n  H2 p < .05 after dropping any single system: "
        f"{(loo['H2_p'] < .05).sum()}/{len(loo)}")
    log(f"  H3 p < .05 after dropping any single system: "
        f"{(loo['H3_p'] < .05).sum()}/{len(loo)}")

    section("S6. WITHIN-SYSTEM ESTIMATES")
    for mdl in d["model_grp"].value_counts().head(3).index:
        sub = d[d["model_grp"] == mdl]
        try:
            r = smf.logit(f"backlash ~ {FOCAL}", data=sub).fit(disp=0, maxiter=200)
            report(r, f"{mdl} (N = {len(sub):,})")
        except Exception as e:
            log(f"  {mdl}: did not converge ({e})")

    section("S7. LANGUAGE")
    for lab, sub in [("English only", d[d["language"] == "English"]),
                     ("non-English only", d[(d["language"] != "English") &
                                            (d["language"] != "unknown")])]:
        if len(sub) < 300:
            log(f"\n  {lab}: N = {len(sub)} - too small")
            continue
        try:
            r = smf.logit(f"backlash ~ {FOCAL}", data=sub).fit(disp=0, maxiter=200)
            report(r, f"{lab} (N = {len(sub):,}, "
                      f"backlash {100 * sub['backlash'].mean():.1f}%)")
        except Exception as e:
            log(f"  {lab}: failed ({e})")

    section("S8. ALTERNATIVE OUTCOME DEFINITIONS")
    d["bl_strict"] = d["user_response_category"].isin(
        ["challenges_refusal", "repeats_original_request"]).astype(int)
    d["bl_challenge"] = (d["user_response_category"] == "challenges_refusal").astype(int)
    log(f"  broad backlash  : {int(d['backlash'].sum()):,} "
        f"({100 * d['backlash'].mean():.1f}%)")
    log(f"  strict backlash : {int(d['bl_strict'].sum()):,} "
        f"({100 * d['bl_strict'].mean():.1f}%)")
    log(f"  challenges only : {int(d['bl_challenge'].sum()):,} "
        f"({100 * d['bl_challenge'].mean():.1f}%)")
    for dv, lab in [("bl_strict", "strict definition"),
                    ("bl_challenge", "challenges refusal only")]:
        r = smf.logit(f"{dv} ~ {FOCAL}", data=d).fit(disp=0, maxiter=200)
        report(r, lab)

    section("S9. SUBGROUP SPLITS ON THE REQUEST CONTROLS")
    if has_tox:
        for lab, sub in [("low toxicity (<= .5)", d[d["tox"] <= .5]),
                         ("high toxicity (> .5)", d[d["tox"] > .5])]:
            if len(sub) < 300:
                log(f"\n  {lab}: N = {len(sub)} - too small")
                continue
            r = smf.logit(f"backlash ~ {FOCAL}", data=sub).fit(disp=0, maxiter=200)
            report(r, f"{lab} (N = {len(sub):,}, "
                      f"backlash {100 * sub['backlash'].mean():.1f}%)")
    if has_sev:
        for lab, sub in [("low severity (0-1)", d[d["sev"] <= 1]),
                         ("moderate severity (2)", d[d["sev"] == 2]),
                         ("high severity (3-4)", d[d["sev"] >= 3])]:
            if len(sub) < 300:
                log(f"\n  {lab}: N = {len(sub)} - too small")
                continue
            r = smf.logit(f"backlash ~ {FOCAL}", data=sub).fit(disp=0, maxiter=200)
            report(r, f"{lab} (N = {len(sub):,}, "
                      f"backlash {100 * sub['backlash'].mean():.1f}%)")

    section("S10. ALTERNATIVE SEVERITY SPECIFICATIONS")
    if has_sev:
        base = f"backlash ~ {FOCAL}"
        r_lin = smf.logit(base + " + sev_c", data=d).fit(disp=0, maxiter=200)
        log(f"\n  severity as linear (0-4): B = {r_lin.params['sev_c']:+.4f}, "
            f"p = {r_lin.pvalues['sev_c']:.4f} {stars(r_lin.pvalues['sev_c'])}")
        r_cat = smf.logit(base + " + C(sev)", data=d).fit(disp=0, maxiter=200)
        log("  severity as categorical (reference = 0):")
        for t in r_cat.params.index:
            if t.startswith("C(sev)"):
                log(f"    {t:22s} B = {r_cat.params[t]:+.4f}, "
                    f"p = {r_cat.pvalues[t]:.4f} {stars(r_cat.pvalues[t])}")
        d["sev_high"] = (d["sev"] >= 3).astype(int)
        r_bin = smf.logit(base + " + sev_high", data=d).fit(disp=0, maxiter=200)
        log(f"  severity as binary (3-4 vs 0-2): B = {r_bin.params['sev_high']:+.4f}, "
            f"p = {r_bin.pvalues['sev_high']:.4f} {stars(r_bin.pvalues['sev_high'])}")
        report(r_lin, "focal terms with linear severity")
    else:
        log("  severity not available")

    section("DONE")
    log(f"Written to {os.path.abspath(args.outdir)}")

    with open(os.path.join(args.outdir, "supplementary_log.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_LOG))


if __name__ == "__main__":
    main()
