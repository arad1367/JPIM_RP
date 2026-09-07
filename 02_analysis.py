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
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

BACKLASH_CATEGORIES = [
    "challenges_refusal", "repeats_original_request",
    "clarification_request", "uncategorized",
]

PAPER_MODEL2 = {
    "disclaimer": 0.112, "framing": 0.331, "apology": -0.270,
    "dis_x_fram": 0.243, "dis_x_apol": -0.260, "fram_x_apol": -0.191,
}

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


def nagelkerke(res, y):
    n = len(y)
    ll1 = res.llf
    ll0 = res.llnull
    cs = 1 - np.exp((2.0 / n) * (ll0 - ll1))
    mx = 1 - np.exp((2.0 / n) * ll0)
    return cs, cs / mx if mx > 0 else np.nan


def fit_logit(df, formula, label, cluster=None):
    if cluster is None:
        res = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
    else:
        res = smf.logit(formula, data=df).fit(
            disp=0, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": df[cluster]})
    y = res.model.endog
    cs, nk = nagelkerke(res, y)
    log(f"\n--- {label} ---")
    if cluster is None:
        log(f"    N = {int(res.nobs):,}   "
            f"chi2({int(res.df_model)}) = {res.llr:.2f}, p = {res.llr_pvalue:.4g}")
    else:
        log(f"    N = {int(res.nobs):,} (clustered SEs)")
    log(f"    -2LL = {-2 * res.llf:,.2f}   Nagelkerke R2 = {nk:.4f}")
    return {"label": label, "res": res, "nagelkerke": nk, "cox_snell": cs,
            "formula": formula, "cluster": cluster}


def coef_table(fits, keep_prefixes=None):
    rows = {}
    for f in fits:
        r = f["res"]
        for name in r.params.index:
            if keep_prefixes and not any(name.startswith(p) for p in keep_prefixes):
                if name != "Intercept":
                    continue
            b = r.params[name]
            se = r.bse[name]
            p = r.pvalues[name]
            rows.setdefault(name, {})[f["label"]] = \
                f"{b:.3f}{stars(p)} ({se:.3f})"
    out = pd.DataFrame(rows).T
    out = out.reindex(columns=[f["label"] for f in fits])

    stat_rows = {}
    for f in fits:
        r = f["res"]
        stat_rows.setdefault("N", {})[f["label"]] = f"{int(r.nobs):,}"
        stat_rows.setdefault("-2 log likelihood", {})[f["label"]] = f"{-2 * r.llf:,.2f}"
        stat_rows.setdefault("Nagelkerke R2", {})[f["label"]] = f"{f['nagelkerke']:.4f}"
        stat_rows.setdefault("Model chi2 (df)", {})[f["label"]] = \
            (f"{r.llr:.2f}{stars(r.llr_pvalue)} ({int(r.df_model)})"
             if f["cluster"] is None else "n/a (clustered)")
    stats_df = pd.DataFrame(stat_rows).T.reindex(
        columns=[f["label"] for f in fits])
    return pd.concat([out, stats_df])


def lr_test(fit_small, fit_big, name):
    r0, r1 = fit_small["res"], fit_big["res"]
    if int(r0.nobs) != int(r1.nobs):
        log(f"  {name}: SKIPPED (different N: "
            f"{int(r0.nobs):,} vs {int(r1.nobs):,})")
        return None
    d = 2 * (r1.llf - r0.llf)
    ddf = int(r1.df_model - r0.df_model)
    p = stats.chi2.sf(d, ddf)
    log(f"  {name}: delta chi2({ddf}) = {d:.2f}, p = {p:.4g} {stars(p)}")
    return {"test": name, "delta_chi2": d, "df": ddf, "p": p}


def margins_2x2(fit, dfin, focal, moderator):
    out = []
    for m in (0, 1):
        for f in (0, 1):
            tmp = dfin.copy()
            tmp[focal] = f
            tmp[moderator] = m
            p = fit["res"].predict(tmp).mean()
            out.append({"moderator": moderator, "moderator_level": m,
                        "focal": focal, "focal_level": f,
                        "pred_backlash_pct": round(100 * p, 1)})
    return pd.DataFrame(out)


def vif_table(df, cols):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = pd.get_dummies(df[cols], drop_first=True).astype(float)
    X = sm.add_constant(X)
    rows = []
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        try:
            rows.append({"variable": c,
                         "VIF": round(variance_inflation_factor(X.values, i), 3)})
        except Exception:
            rows.append({"variable": c, "VIF": np.nan})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--no-toxicity", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    section("0. LOAD AND BUILD THE ANALYSIS SAMPLE")
    df = pd.read_csv(args.input, low_memory=False)
    log(f"  loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    ref = df["is_actual_refusal"].astype(str).str.strip().str.upper()
    d = df[ref == "TRUE"].copy().reset_index(drop=True)
    log(f"  confirmed refusals: {len(d):,}")

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

    has_tox = ("toxicity_max" in d.columns) and \
              d["toxicity_max"].notna().any() and not args.no_toxicity
    log(f"  toxicity available: {has_tox}")
    if has_tox:
        d["tox_max"] = d["toxicity_max"].fillna(d["toxicity_max"].median())
        d["tox_max_c"] = d["tox_max"] - d["tox_max"].mean()
        d["tox_high"] = (d["tox_max"] > 0.5).astype(int)
        log(f"  toxicity_max: mean {d['tox_max'].mean():.4f}, "
            f"p90 {d['tox_max'].quantile(.90):.4f}, "
            f"share > .5 = {100 * d['tox_high'].mean():.1f}%")

    top_topic = d["topic"].value_counts().idxmax()
    top_model = d["model_grp"].value_counts().idxmax()
    log(f"  reference topic = {top_topic}; reference model = {top_model}")

    section("PART A - CURRENT MANUSCRIPT (WASKO)")

    log("\nTable 2. Unconditional backlash rates")
    t2 = []
    for lab, col in [("AI identity disclaimer", "disclaimer"),
                     ("Normative framing", "framing"),
                     ("Apology", "apology")]:
        pres = d.loc[d[col] == 1, "backlash"].mean()
        absent = d.loc[d[col] == 0, "backlash"].mean()
        t2.append({"feature": lab, "N_present": int(d[col].sum()),
                   "pct_present": round(100 * d[col].mean(), 1),
                   "backlash_present_pct": round(100 * pres, 1),
                   "backlash_absent_pct": round(100 * absent, 1),
                   "delta_pp": round(100 * (pres - absent), 1)})
    t2 = pd.DataFrame(t2)
    log(t2.to_string(index=False))
    t2.to_csv(os.path.join(args.outdir, "table2_unconditional.csv"), index=False)

    log(f"\n  overall backlash: {int(d['backlash'].sum()):,} "
        f"({100 * d['backlash'].mean():.1f}%)")
    log(f"  co-occurrence  disclaimer & apology : "
        f"{int(((d.disclaimer == 1) & (d.apology == 1)).sum()):,}")
    log(f"  co-occurrence  framing & apology    : "
        f"{int(((d.framing == 1) & (d.apology == 1)).sum()):,}")
    log(f"  co-occurrence  disclaimer & framing : "
        f"{int(((d.disclaimer == 1) & (d.framing == 1)).sum()):,}")
    log(f"  all three                           : "
        f"{int(((d.disclaimer == 1) & (d.framing == 1) & (d.apology == 1)).sum()):,}")
    log(f"  none of the three                   : "
        f"{int(((d.disclaimer == 0) & (d.framing == 0) & (d.apology == 0)).sum()):,}")

    m1 = fit_logit(d, "backlash ~ disclaimer + framing + apology", "Model 1")
    m2 = fit_logit(d, "backlash ~ disclaimer * framing + disclaimer * apology "
                      "+ framing * apology", "Model 2")
    d3 = d.dropna(subset=["english"]).copy()
    m3 = fit_logit(d3, "backlash ~ disclaimer * framing + disclaimer * apology "
                       "+ framing * apology + turn + english", "Model 3")

    log("\nTable 3. Main models")
    t3 = coef_table([m1, m2, m3])
    log(t3.to_string())
    t3.to_csv(os.path.join(args.outdir, "table3_main_models.csv"))

    log("\nReproduction check against the manuscript (Model 2):")
    got = {
        "disclaimer": m2["res"].params.get("disclaimer", np.nan),
        "framing": m2["res"].params.get("framing", np.nan),
        "apology": m2["res"].params.get("apology", np.nan),
        "dis_x_fram": m2["res"].params.get("disclaimer:framing", np.nan),
        "dis_x_apol": m2["res"].params.get("disclaimer:apology", np.nan),
        "fram_x_apol": m2["res"].params.get("framing:apology", np.nan),
    }
    ok_all = True
    for k, v in got.items():
        exp = PAPER_MODEL2[k]
        ok = abs(v - exp) < 0.005
        ok_all &= ok
        log(f"    {'OK ' if ok else 'XX '} {k:12s} got {v:+.3f}  paper {exp:+.3f}")
    log("  --> Table 3 reproduces." if ok_all else
        "  --> MISMATCH: check variable construction before writing up.")

    log("\nLikelihood-ratio tests:")
    lr_test(m1, m2, "Model 1 -> Model 2 (adding interactions)")

    log("\nFigure 2. Predictive margins from Model 2")
    marg = pd.concat([
        margins_2x2(m2, d, "disclaimer", "framing"),
        margins_2x2(m2, d, "disclaimer", "apology"),
    ], ignore_index=True)
    log(marg.to_string(index=False))
    marg.to_csv(os.path.join(args.outdir, "figure2_margins.csv"), index=False)

    section("PART B - ROBUSTNESS WITH ADDITIONAL CONTROLS (JOHANNES)")

    log("\nB0. Descriptives by model")
    by_model = d.groupby("model_grp").agg(
        n=("backlash", "size"),
        backlash_pct=("backlash", lambda s: round(100 * s.mean(), 1)),
        disclaimer_pct=("disclaimer", lambda s: round(100 * s.mean(), 1)),
        framing_pct=("framing", lambda s: round(100 * s.mean(), 1)),
        apology_pct=("apology", lambda s: round(100 * s.mean(), 1)),
    ).sort_values("n", ascending=False)
    if has_tox:
        by_model["tox_max_mean"] = d.groupby("model_grp")["tox_max"].mean().round(4)
    log(by_model.to_string())
    by_model.to_csv(os.path.join(args.outdir, "descriptives_by_model.csv"))

    log("\n  Chi-square: is apology use independent of model?")
    ct = pd.crosstab(d["model_grp"], d["apology"])
    c2, p, dof, _ = stats.chi2_contingency(ct)
    log(f"    chi2({dof}) = {c2:.1f}, p = {p:.3g}  "
        f"-> {'models differ strongly' if p < .001 else 'no clear difference'}")

    log("\n  Backlash rate by request topic:")
    by_topic = d.groupby("topic").agg(
        n=("backlash", "size"),
        backlash_pct=("backlash", lambda s: round(100 * s.mean(), 1)),
        framing_pct=("framing", lambda s: round(100 * s.mean(), 1)),
    ).sort_values("n", ascending=False)
    log(by_topic.to_string())
    by_topic.to_csv(os.path.join(args.outdir, "descriptives_by_topic.csv"))

    base_int = "disclaimer * framing + disclaimer * apology + framing * apology"
    T = f"C(topic, Treatment('{top_topic}'))"
    M = f"C(model_grp, Treatment('{top_model}'))"

    fits_b = []
    m4 = fit_logit(d, f"backlash ~ {base_int} + {T}", "Model 4 (+topic)")
    fits_b.append(m4)

    if has_tox:
        m5 = fit_logit(d, f"backlash ~ {base_int} + tox_max_c",
                       "Model 5 (+toxicity)")
        fits_b.append(m5)

    m6 = fit_logit(d, f"backlash ~ {base_int} + {M}", "Model 6 (+model FE)")
    fits_b.append(m6)

    full = f"backlash ~ {base_int} + turn + english + {T} + {M}"
    if has_tox:
        full += " + tox_max_c"
    m7 = fit_logit(d3, full, "Model 7 (full)")
    fits_b.append(m7)

    m8 = fit_logit(d3, full, "Model 8 (clustered)", cluster="model_grp")
    fits_b.append(m8)

    log("\nTable 4. Robustness models (focal terms only)")
    t4 = coef_table(fits_b, keep_prefixes=["disclaimer", "framing", "apology",
                                           "turn", "english", "tox_"])
    log(t4.to_string())
    t4.to_csv(os.path.join(args.outdir, "table4_robustness.csv"))

    log("\nFull coefficients including all dummies (Model 7):")
    m7full = pd.DataFrame({
        "B": m7["res"].params.round(4),
        "SE": m7["res"].bse.round(4),
        "z": m7["res"].tvalues.round(3),
        "p": m7["res"].pvalues.round(4),
    })
    m7full["sig"] = m7full["p"].apply(stars)
    log(m7full.to_string())
    m7full.to_csv(os.path.join(args.outdir, "model7_full_coefficients.csv"))

    section("B1. DO THE FOCAL INTERACTIONS SURVIVE?")
    verdict = []
    for f in [m2, m3] + fits_b:
        r = f["res"]
        row = {"model": f["label"]}
        for term, hyp in [("disclaimer:framing", "H2"),
                          ("disclaimer:apology", "H3")]:
            if term in r.params.index:
                row[f"{hyp}_B"] = round(r.params[term], 3)
                row[f"{hyp}_SE"] = round(r.bse[term], 3)
                row[f"{hyp}_p"] = round(r.pvalues[term], 4)
                row[f"{hyp}_sig"] = "yes" if r.pvalues[term] < .05 else "no"
        verdict.append(row)
    verdict = pd.DataFrame(verdict)
    log(verdict.to_string(index=False))
    verdict.to_csv(os.path.join(args.outdir, "focal_interactions_across_models.csv"),
                   index=False)

    n_h2 = (verdict.get("H2_sig") == "yes").sum()
    n_h3 = (verdict.get("H3_sig") == "yes").sum()
    log(f"\n  H2 significant in {n_h2}/{len(verdict)} specifications")
    log(f"  H3 significant in {n_h3}/{len(verdict)} specifications")

    section("B2. DIAGNOSTICS")
    log("\nLikelihood-ratio tests (common samples only):")
    diag = [x for x in [
        lr_test(m2, m4, "Model 2 -> Model 4 (topic)"),
        lr_test(m2, m6, "Model 2 -> Model 6 (model FE)"),
    ] if x]
    if has_tox:
        x = lr_test(m2, m5, "Model 2 -> Model 5 (toxicity)")
        if x:
            diag.append(x)

    log("\nVariance inflation factors (Model 7 predictors):")
    vcols = ["disclaimer", "framing", "apology", "turn", "topic", "model_grp"]
    if has_tox:
        vcols.append("tox_max_c")
    v = vif_table(d3, vcols)
    log(v.head(20).to_string(index=False))
    v.to_csv(os.path.join(args.outdir, "vif.csv"), index=False)
    if (v["VIF"] > 10).any():
        log("  some VIF above 10 - report and discuss")
    else:
        log("  all VIF below 10")

    log("\nModel-fit comparison:")
    fitcmp = pd.DataFrame([{
        "model": f["label"], "N": int(f["res"].nobs),
        "-2LL": round(-2 * f["res"].llf, 2),
        "Nagelkerke_R2": round(f["nagelkerke"], 4),
        "AIC": round(f["res"].aic, 2), "BIC": round(f["res"].bic, 2),
    } for f in [m1, m2, m3] + fits_b])
    log(fitcmp.to_string(index=False))
    fitcmp.to_csv(os.path.join(args.outdir, "model_diagnostics.csv"), index=False)

    section("B3. JOHANNES'S ALTERNATIVE EXPLANATION")
    log("Claim: normative refusals may just come from weaker models, and users")
    log("are annoyed by the weak model, not by the moralising.")
    log("If true, the framing effect should shrink towards zero once model")
    log("fixed effects absorb between-model quality differences.")
    b_m2 = m2["res"].params.get("framing", np.nan)
    b_m6 = m6["res"].params.get("framing", np.nan)
    log(f"\n  framing main effect, Model 2 (no model FE): B = {b_m2:+.3f}")
    log(f"  framing main effect, Model 6 (+model FE)  : B = {b_m6:+.3f}")
    log(f"  change: {100 * (b_m6 - b_m2) / abs(b_m2):+.1f}%")

    log("\nWithin-model check: the two largest models estimated separately")
    for mdl in d["model_grp"].value_counts().head(2).index:
        sub = d[d["model_grp"] == mdl]
        try:
            r = smf.logit(f"backlash ~ {base_int}", data=sub).fit(disp=0, maxiter=200)
            log(f"\n  {mdl} (N = {len(sub):,}):")
            for t, h in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
                if t in r.params.index:
                    log(f"    {h}: B = {r.params[t]:+.3f}, "
                        f"SE = {r.bse[t]:.3f}, p = {r.pvalues[t]:.3f} "
                        f"{stars(r.pvalues[t])}")
        except Exception as e:
            log(f"  {mdl}: model did not converge ({e})")

    if has_tox:
        log("\nToxicity split: does the pattern hold for benign vs toxic requests?")
        for lab, sub in [("low toxicity (<= .5)", d[d["tox_high"] == 0]),
                         ("high toxicity (> .5)", d[d["tox_high"] == 1])]:
            if len(sub) < 300:
                log(f"  {lab}: N = {len(sub)} - too small, skipped")
                continue
            try:
                r = smf.logit(f"backlash ~ {base_int}", data=sub).fit(disp=0, maxiter=200)
                log(f"\n  {lab} (N = {len(sub):,}, "
                    f"backlash {100 * sub['backlash'].mean():.1f}%):")
                for t, h in [("disclaimer:framing", "H2"),
                             ("disclaimer:apology", "H3")]:
                    if t in r.params.index:
                        log(f"    {h}: B = {r.params[t]:+.3f}, "
                            f"p = {r.pvalues[t]:.3f} {stars(r.pvalues[t])}")
            except Exception as e:
                log(f"  {lab}: did not converge ({e})")

    log("\nPredictive margins from Model 7 (full controls):")
    marg7 = pd.concat([
        margins_2x2(m7, d3, "disclaimer", "framing"),
        margins_2x2(m7, d3, "disclaimer", "apology"),
    ], ignore_index=True)
    log(marg7.to_string(index=False))
    marg7.to_csv(os.path.join(args.outdir, "figure2_margins_model7.csv"), index=False)

    if "sentiment_after_refusal" in d.columns and \
            d["sentiment_after_refusal"].notna().any():
        section("B4. SENTIMENT AS A SECONDARY OUTCOME")
        s = d.dropna(subset=["sentiment_after_refusal"])
        log(f"  N with sentiment = {len(s):,}, "
            f"mean = {s['sentiment_after_refusal'].mean():.4f}")
        log(f"  sentiment | backlash=1 : "
            f"{s.loc[s.backlash == 1, 'sentiment_after_refusal'].mean():.4f}")
        log(f"  sentiment | backlash=0 : "
            f"{s.loc[s.backlash == 0, 'sentiment_after_refusal'].mean():.4f}")
        t, p = stats.ttest_ind(
            s.loc[s.backlash == 1, "sentiment_after_refusal"],
            s.loc[s.backlash == 0, "sentiment_after_refusal"], equal_var=False)
        log(f"  Welch t = {t:.2f}, p = {p:.3g}")
        r = smf.ols(f"sentiment_after_refusal ~ {base_int}", data=s).fit()
        log("\n  OLS on sentiment, focal terms:")
        for t_ in ["disclaimer", "framing", "apology",
                   "disclaimer:framing", "disclaimer:apology"]:
            if t_ in r.params.index:
                log(f"    {t_:22s} b = {r.params[t_]:+.4f}, "
                    f"p = {r.pvalues[t_]:.3f} {stars(r.pvalues[t_])}")

    section("DONE")
    log(f"All tables written to: {os.path.abspath(args.outdir)}")
    log("analysis_log.txt plus the CSVs are ready.")

    with open(os.path.join(args.outdir, "analysis_log.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_LOG))


if __name__ == "__main__":
    main()
