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
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)

BASE = "/content/drive/MyDrive/Wasko_JPIM"
DEFAULT_IN = f"{BASE}/data/Wasko_Final_Dataset_ANALYSIS.csv"
DEFAULT_OUT = f"{BASE}/output/results"

BACKLASH = ["challenges_refusal", "repeats_original_request", "clarification_request"]
FOCAL = "disclaimer * framing + disclaimer * apology + framing * apology"

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


def nagelkerke(res):
    n = len(res.model.endog)
    cs = 1 - np.exp((2.0 / n) * (res.llnull - res.llf))
    mx = 1 - np.exp((2.0 / n) * res.llnull)
    return cs / mx if mx > 0 else np.nan


def fit(df, formula, label, cluster=None):
    if cluster is None:
        res = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
    else:
        res = smf.logit(formula, data=df).fit(
            disp=0, maxiter=200, cov_type="cluster",
            cov_kwds={"groups": df[cluster]})
    log(f"\n--- {label} ---")
    if cluster is None:
        log(f"    N = {int(res.nobs):,}   chi2({int(res.df_model)}) = {res.llr:.2f}, "
            f"p = {res.llr_pvalue:.4g}   Nagelkerke R2 = {nagelkerke(res):.4f}")
    else:
        log(f"    N = {int(res.nobs):,}   clustered SEs   "
            f"Nagelkerke R2 = {nagelkerke(res):.4f}")
    return {"label": label, "res": res, "nk": nagelkerke(res), "cluster": cluster}


def coef_table(fits, keep=None):
    rows = {}
    for f in fits:
        r = f["res"]
        for name in r.params.index:
            if keep and not any(name.startswith(k) for k in keep) and name != "Intercept":
                continue
            rows.setdefault(name, {})[f["label"]] = \
                f"{r.params[name]:.3f}{stars(r.pvalues[name])} ({r.bse[name]:.3f})"
    out = pd.DataFrame(rows).T.reindex(columns=[f["label"] for f in fits])
    stat = {}
    for f in fits:
        r = f["res"]
        stat.setdefault("N", {})[f["label"]] = f"{int(r.nobs):,}"
        stat.setdefault("-2 log likelihood", {})[f["label"]] = f"{-2 * r.llf:,.2f}"
        stat.setdefault("Nagelkerke R2", {})[f["label"]] = f"{f['nk']:.4f}"
        stat.setdefault("Model chi2 (df)", {})[f["label"]] = (
            f"{r.llr:.2f}{stars(r.llr_pvalue)} ({int(r.df_model)})"
            if f["cluster"] is None else "n/a")
    stat_df = pd.DataFrame(stat).T.reindex(columns=[f["label"] for f in fits])
    return pd.concat([out, stat_df])


def lr_test(small, big, name):
    r0, r1 = small["res"], big["res"]
    if int(r0.nobs) != int(r1.nobs):
        log(f"  {name}: skipped (different N)")
        return
    d = 2 * (r1.llf - r0.llf)
    ddf = int(r1.df_model - r0.df_model)
    p = stats.chi2.sf(d, ddf)
    log(f"  {name}: delta chi2({ddf}) = {d:.2f}, p = {p:.4g} {stars(p)}")


def margins(f, data, focal, moderator):
    out = []
    for m in (0, 1):
        for x in (0, 1):
            tmp = data.copy()
            tmp[focal] = x
            tmp[moderator] = m
            out.append({"moderator": moderator, "moderator_level": m,
                        "focal": focal, "focal_level": x,
                        "pred_backlash_pct": round(100 * f["res"].predict(tmp).mean(), 1)})
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
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    section("0. ANALYSIS SAMPLE")
    raw = pd.read_csv(args.input, low_memory=False)
    log(f"  loaded {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    d = raw[raw["is_actual_refusal"].astype(str).str.strip().str.upper() == "TRUE"].copy()
    d = d.reset_index(drop=True)
    log(f"  confirmed refusals: {len(d):,}")

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

    has_tox = "toxicity_max" in d.columns and d["toxicity_max"].notna().any()
    has_sev = "request_severity" in d.columns and d["request_severity"].notna().any()
    has_sent = "sentiment_after_refusal" in d.columns and d["sentiment_after_refusal"].notna().any()
    log(f"  toxicity: {has_tox}   severity: {has_sev}   sentiment: {has_sent}")

    if has_tox:
        d["tox"] = d["toxicity_max"].fillna(d["toxicity_max"].median())
        d["tox_c"] = d["tox"] - d["tox"].mean()
    if has_sev:
        d["sev"] = d["request_severity"].fillna(d["request_severity"].median())
        d["sev_c"] = d["sev"] - d["sev"].mean()
        log(f"  severity mean = {d['sev'].mean():.3f}, "
            f"share 3-4 = {100 * (d['sev'] >= 3).mean():.1f}%")

    top_topic = d["topic"].value_counts().idxmax()
    top_model = d["model_grp"].value_counts().idxmax()
    T = f"C(topic, Treatment('{top_topic}'))"
    M = f"C(model_grp, Treatment('{top_model}'))"
    dc = d.dropna(subset=["english"]).copy()
    log(f"  reference topic = {top_topic}; reference system = {top_model}")
    log(f"  complete cases with language identified: {len(dc):,}")

    section("TABLE 2. UNCONDITIONAL BACKLASH RATES")
    rows = []
    for lab, col in [("AI identity disclaimer", "disclaimer"),
                     ("Normative framing", "framing"),
                     ("Apology", "apology")]:
        pres = d.loc[d[col] == 1, "backlash"].mean()
        absent = d.loc[d[col] == 0, "backlash"].mean()
        rows.append({"feature": lab, "N_present": int(d[col].sum()),
                     "pct_present": round(100 * d[col].mean(), 1),
                     "backlash_present_pct": round(100 * pres, 1),
                     "backlash_absent_pct": round(100 * absent, 1),
                     "delta_pp": round(100 * (pres - absent), 1)})
    t2 = pd.DataFrame(rows)
    log(t2.to_string(index=False))
    t2.to_csv(os.path.join(args.outdir, "table2_unconditional.csv"), index=False)

    log(f"\n  overall backlash: {int(d['backlash'].sum()):,} "
        f"({100 * d['backlash'].mean():.1f}%)")
    for lab, mask in [
        ("disclaimer & apology", (d.disclaimer == 1) & (d.apology == 1)),
        ("framing & apology", (d.framing == 1) & (d.apology == 1)),
        ("disclaimer & framing", (d.disclaimer == 1) & (d.framing == 1)),
        ("all three", (d.disclaimer == 1) & (d.framing == 1) & (d.apology == 1)),
        ("none of the three", (d.disclaimer == 0) & (d.framing == 0) & (d.apology == 0)),
    ]:
        log(f"  {lab:24s} {int(mask.sum()):,}")

    log("\n  backlash components:")
    for cat in BACKLASH:
        n = int((d["user_response_category"] == cat).sum())
        log(f"    {cat:26s} {n:6,d}  ({100 * n / len(d):4.1f}%)")
    for cat in ["topic_shift", "compliant_reformulation", "takes_alternative",
                "accepts_boundary", "abandonment"]:
        n = int((d["user_response_category"] == cat).sum())
        log(f"    {cat:26s} {n:6,d}  ({100 * n / len(d):4.1f}%)")

    section("TABLE 3. MAIN MODELS")
    m1 = fit(d, "backlash ~ disclaimer + framing + apology", "Model 1")
    m2 = fit(d, f"backlash ~ {FOCAL}", "Model 2")
    m3 = fit(dc, f"backlash ~ {FOCAL} + turn + english", "Model 3")

    t3 = coef_table([m1, m2, m3])
    log("")
    log(t3.to_string())
    t3.to_csv(os.path.join(args.outdir, "table3_main_models.csv"))

    log("\nLikelihood-ratio test:")
    lr_test(m1, m2, "Model 1 -> Model 2 (adding interactions)")

    log("\nFocal hypothesis tests:")
    r1 = m1["res"]
    w1 = (r1.params["disclaimer"] / r1.bse["disclaimer"]) ** 2
    log(f"  H1 (Model 1, main effect): B = {r1.params['disclaimer']:+.3f}, "
        f"SE = {r1.bse['disclaimer']:.3f}, Wald chi2(1) = {w1:.2f}, "
        f"p = {r1.pvalues['disclaimer']:.4f} {stars(r1.pvalues['disclaimer'])}")
    for term, hyp in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
        r = m2["res"]
        w = (r.params[term] / r.bse[term]) ** 2
        log(f"  {hyp} (Model 2, interaction): B = {r.params[term]:+.3f}, "
            f"SE = {r.bse[term]:.3f}, Wald chi2(1) = {w:.2f}, "
            f"p = {r.pvalues[term]:.4f} {stars(r.pvalues[term])}")

    section("FIGURE 2. PREDICTIVE MARGINS FROM MODEL 2")
    marg = pd.concat([margins(m2, d, "disclaimer", "framing"),
                      margins(m2, d, "disclaimer", "apology")], ignore_index=True)
    log(marg.to_string(index=False))
    marg.to_csv(os.path.join(args.outdir, "figure2_margins.csv"), index=False)

    section("TABLE 4. ROBUSTNESS WITH CONTROLS")
    fits_r = []
    m4 = fit(d, f"backlash ~ {FOCAL} + {T}", "Model 4 (+topic)")
    fits_r.append(m4)
    if has_tox:
        m5 = fit(d, f"backlash ~ {FOCAL} + tox_c", "Model 5 (+toxicity)")
        fits_r.append(m5)
    if has_sev:
        m6 = fit(d, f"backlash ~ {FOCAL} + sev_c", "Model 6 (+severity)")
        fits_r.append(m6)
    m7 = fit(d, f"backlash ~ {FOCAL} + {M}", "Model 7 (+system FE)")
    fits_r.append(m7)

    full = f"backlash ~ {FOCAL} + turn + english + {T}"
    if has_tox:
        full += " + tox_c"
    if has_sev:
        full += " + sev_c"
    full += f" + {M}"
    m8 = fit(dc, full, "Model 8 (full)")
    fits_r.append(m8)

    t4 = coef_table(fits_r, keep=["disclaimer", "framing", "apology",
                                  "turn", "english", "tox_", "sev_"])
    log("")
    log(t4.to_string())
    t4.to_csv(os.path.join(args.outdir, "table4_robustness.csv"))

    log("\nFull coefficients of Model 8 including all dummies:")
    full_tab = pd.DataFrame({
        "B": m8["res"].params.round(4), "SE": m8["res"].bse.round(4),
        "z": m8["res"].tvalues.round(3), "p": m8["res"].pvalues.round(4)})
    full_tab["sig"] = full_tab["p"].apply(stars)
    log(full_tab.to_string())
    full_tab.to_csv(os.path.join(args.outdir, "model8_full_coefficients.csv"))

    log("\nModel 8 re-estimated with standard errors clustered by responding system:")
    m8c = fit(dc, full, "Model 8 clustered", cluster="model_grp")
    for term, hyp in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
        r = m8c["res"]
        log(f"  {hyp}: B = {r.params[term]:+.3f}, SE = {r.bse[term]:.3f}, "
            f"p = {r.pvalues[term]:.4f} {stars(r.pvalues[term])}")

    section("FOCAL INTERACTIONS ACROSS ALL SPECIFICATIONS")
    verdict = []
    for f in [m2, m3] + fits_r + [m8c]:
        r = f["res"]
        row = {"model": f["label"]}
        for term, hyp in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
            if term in r.params.index:
                row[f"{hyp}_B"] = round(r.params[term], 3)
                row[f"{hyp}_SE"] = round(r.bse[term], 3)
                row[f"{hyp}_p"] = round(r.pvalues[term], 4)
                row[f"{hyp}_sig"] = "yes" if r.pvalues[term] < .05 else "no"
        verdict.append(row)
    verdict = pd.DataFrame(verdict)
    log(verdict.to_string(index=False))
    verdict.to_csv(os.path.join(args.outdir, "focal_interactions.csv"), index=False)
    log(f"\n  H2 significant in {(verdict['H2_sig'] == 'yes').sum()}/{len(verdict)}")
    log(f"  H3 significant in {(verdict['H3_sig'] == 'yes').sum()}/{len(verdict)}")

    section("DIAGNOSTICS")
    log("Likelihood-ratio tests against Model 2:")
    lr_test(m2, m4, "+ request topic")
    if has_tox:
        lr_test(m2, m5, "+ request toxicity")
    if has_sev:
        lr_test(m2, m6, "+ request severity")
    lr_test(m2, m7, "+ system fixed effects")

    log("\nVariance inflation factors for Model 8:")
    vcols = ["disclaimer", "framing", "apology", "turn", "topic", "model_grp"]
    if has_tox:
        vcols.append("tox_c")
    if has_sev:
        vcols.append("sev_c")
    v = vif_table(dc, vcols)
    log(v.head(25).to_string(index=False))
    v.to_csv(os.path.join(args.outdir, "vif.csv"), index=False)
    if has_sev:
        sev_vif = v.loc[v["variable"] == "sev_c", "VIF"]
        if len(sev_vif):
            log(f"\n  severity VIF = {float(sev_vif.iloc[0]):.3f}")
            log("  a value near 1 means severity is not redundant with topic")

    log("\nModel fit comparison:")
    cmp = pd.DataFrame([{
        "model": f["label"], "N": int(f["res"].nobs),
        "-2LL": round(-2 * f["res"].llf, 2),
        "Nagelkerke_R2": round(f["nk"], 4),
        "AIC": round(f["res"].aic, 2), "BIC": round(f["res"].bic, 2),
    } for f in [m1, m2, m3] + fits_r])
    log(cmp.to_string(index=False))
    cmp.to_csv(os.path.join(args.outdir, "model_fit.csv"), index=False)

    section("DESCRIPTIVES BY RESPONDING SYSTEM AND TOPIC")
    agg = {"n": ("backlash", "size"),
           "backlash_pct": ("backlash", lambda s: round(100 * s.mean(), 1)),
           "disclaimer_pct": ("disclaimer", lambda s: round(100 * s.mean(), 1)),
           "framing_pct": ("framing", lambda s: round(100 * s.mean(), 1)),
           "apology_pct": ("apology", lambda s: round(100 * s.mean(), 1))}
    by_model = d.groupby("model_grp").agg(**agg).sort_values("n", ascending=False)
    if has_tox:
        by_model["tox_mean"] = d.groupby("model_grp")["tox"].mean().round(4)
    if has_sev:
        by_model["sev_mean"] = d.groupby("model_grp")["sev"].mean().round(3)
    log(by_model.to_string())
    by_model.to_csv(os.path.join(args.outdir, "descriptives_by_system.csv"))

    ct = pd.crosstab(d["model_grp"], d["apology"])
    c2, p, dof, _ = stats.chi2_contingency(ct)
    log(f"\n  apology use by system: chi2({dof}) = {c2:.1f}, p = {p:.3g}")

    by_topic = d.groupby("topic").agg(
        n=("backlash", "size"),
        backlash_pct=("backlash", lambda s: round(100 * s.mean(), 1)),
        framing_pct=("framing", lambda s: round(100 * s.mean(), 1)),
    ).sort_values("n", ascending=False)
    if has_tox:
        by_topic["tox_mean"] = d.groupby("topic")["tox"].mean().round(4)
    if has_sev:
        by_topic["sev_mean"] = d.groupby("topic")["sev"].mean().round(3)
    log("")
    log(by_topic.to_string())
    by_topic.to_csv(os.path.join(args.outdir, "descriptives_by_topic.csv"))

    if has_sent:
        section("TABLE 5. FOLLOW-UP SENTIMENT AS A SECONDARY OUTCOME")
        s = d.dropna(subset=["sentiment_after_refusal"]).copy()
        log(f"  N = {len(s):,}, mean = {s['sentiment_after_refusal'].mean():.4f}, "
            f"SD = {s['sentiment_after_refusal'].std():.4f}")
        a = s.loc[s.backlash == 1, "sentiment_after_refusal"]
        b = s.loc[s.backlash == 0, "sentiment_after_refusal"]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        log(f"  sentiment | backlash = 1 : {a.mean():.4f}")
        log(f"  sentiment | backlash = 0 : {b.mean():.4f}")
        log(f"  Welch t = {t:.2f}, p = {p:.3g}")

        r_main = smf.ols("sentiment_after_refusal ~ disclaimer + framing + apology",
                         data=s).fit()
        log("\n  main-effects model:")
        for term in ["disclaimer", "framing", "apology"]:
            log(f"    {term:12s} b = {r_main.params[term]:+.4f}  "
                f"SE = {r_main.bse[term]:.4f}  p = {r_main.pvalues[term]:.4f} "
                f"{stars(r_main.pvalues[term])}")

        log("\n  pairwise equality tests between coefficients:")
        pairs = [("framing", "disclaimer"), ("framing", "apology"),
                 ("disclaimer", "apology")]
        prows = []
        for x, y in pairs:
            tt = r_main.t_test(f"{x} - {y} = 0")
            pv = float(np.ravel(tt.pvalue)[0])
            log(f"    {x} vs {y}: difference = {float(tt.effect[0]):+.4f}, "
                f"p = {pv:.4f} {stars(pv)}")
            prows.append({"comparison": f"{x} vs {y}",
                          "difference": round(float(tt.effect[0]), 4),
                          "p": round(pv, 4)})
        pd.DataFrame(prows).to_csv(
            os.path.join(args.outdir, "sentiment_coefficient_tests.csv"), index=False)

        sc = s.dropna(subset=["english"]).copy()
        f_sent = f"sentiment_after_refusal ~ {FOCAL} + turn + english + {T}"
        if has_tox:
            f_sent += " + tox_c"
        if has_sev:
            f_sent += " + sev_c"
        r_full = smf.ols(f_sent, data=sc).fit()
        log(f"\n  full specification, N = {int(r_full.nobs):,}, "
            f"adj R2 = {r_full.rsquared_adj:.4f}")
        keep = ["disclaimer", "framing", "apology", "disclaimer:framing",
                "disclaimer:apology", "framing:apology", "tox_c", "sev_c",
                "turn", "english"]
        srows = []
        for term in keep:
            if term in r_full.params.index:
                log(f"    {term:20s} b = {r_full.params[term]:+.4f}  "
                    f"SE = {r_full.bse[term]:.4f}  p = {r_full.pvalues[term]:.4f} "
                    f"{stars(r_full.pvalues[term])}")
                srows.append({"term": term, "B": round(r_full.params[term], 4),
                              "SE": round(r_full.bse[term], 4),
                              "p": round(r_full.pvalues[term], 4),
                              "sig": stars(r_full.pvalues[term])})
        pd.DataFrame(srows).to_csv(
            os.path.join(args.outdir, "table5_sentiment.csv"), index=False)

        r_cl = smf.ols(f_sent, data=sc).fit(
            cov_type="cluster", cov_kwds={"groups": sc["model_grp"]})
        log("\n  same model with standard errors clustered by responding system:")
        for term in ["disclaimer:framing", "disclaimer:apology"]:
            if term in r_cl.params.index:
                log(f"    {term:20s} b = {r_cl.params[term]:+.4f}  "
                    f"p = {r_cl.pvalues[term]:.4f} {stars(r_cl.pvalues[term])}")

    section("DONE")
    log(f"Tables written to {os.path.abspath(args.outdir)}")
    log("Next: 06_supplementary.py, then 07_figure2.py")

    with open(os.path.join(args.outdir, "analysis_log.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_LOG))


if __name__ == "__main__":
    main()
