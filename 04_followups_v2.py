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

BACKLASH = ["challenges_refusal", "repeats_original_request",
            "clarification_request", "uncategorized"]
BASE = "disclaimer * framing + disclaimer * apology + framing * apology"

_LOG = []


def log(m=""):
    print(m, flush=True)
    _LOG.append(str(m))


def section(t):
    log("\n" + "=" * 78)
    log(t)
    log("=" * 78)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "+" if p < .10 else ""


def nagelkerke(res):
    n = len(res.model.endog)
    cs = 1 - np.exp((2.0 / n) * (res.llnull - res.llf))
    mx = 1 - np.exp((2.0 / n) * res.llnull)
    return cs / mx if mx > 0 else np.nan


def show(res, terms, label, logit=True):
    log(f"\n--- {label} ---")
    if logit:
        log(f"    N = {int(res.nobs):,}   Nagelkerke R2 = {nagelkerke(res):.4f}")
    else:
        log(f"    N = {int(res.nobs):,}   adj R2 = {res.rsquared_adj:.4f}")
    for t in terms:
        if t in res.params.index:
            log(f"    {t:24s} b = {res.params[t]:+.4f}  SE = {res.bse[t]:.4f}  "
                f"p = {res.pvalues[t]:.4f} {stars(res.pvalues[t])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/content/drive/MyDrive/Wasko_JPIM/output_v2/Wasko_Final_Dataset_ENRICHED_v2.csv")
    ap.add_argument("--outdir", default="/content/drive/MyDrive/Wasko_JPIM/output_v2/results_followup")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    section("0. SAMPLE")
    df = pd.read_csv(args.input, low_memory=False)
    d = df[df["is_actual_refusal"].astype(str).str.strip().str.upper() == "TRUE"].copy()
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
    has_sent = ("sentiment_after_refusal" in d.columns
                and d["sentiment_after_refusal"].notna().any())
    log(f"  toxicity available: {has_tox}   sentiment available: {has_sent}")
    if has_tox:
        d["tox"] = d["toxicity_max"].fillna(d["toxicity_max"].median())
        d["tox_c"] = d["tox"] - d["tox"].mean()

    top_topic = d["topic"].value_counts().idxmax()
    T = f"C(topic, Treatment('{top_topic}'))"
    dc = d.dropna(subset=["english"]).copy()

    # =====================================================================
    section("Q1 (WASKO). WHAT SHOULD THE MAIN MODEL BE?")
    log("Candidate: focal interactions + episode-level controls only")
    log("(turn, language, request topic, request toxicity), WITHOUT model")
    log("fixed effects. Model fixed effects stay in robustness because they")
    log("are where the H2 identification problem lives.")

    f_main = f"backlash ~ {BASE} + turn + english + {T}"
    if has_tox:
        f_main += " + tox_c"
    m_main = smf.logit(f_main, data=dc).fit(disp=0, maxiter=200)
    show(m_main, ["disclaimer", "framing", "apology", "disclaimer:framing",
                  "disclaimer:apology", "framing:apology", "tox_c", "turn", "english"],
         "PROPOSED MAIN MODEL (episode-level controls, no model FE)")

    m2 = smf.logit(f"backlash ~ {BASE}", data=d).fit(disp=0, maxiter=200)
    log(f"\n  For comparison, Model 2 (no controls):")
    log(f"    H2  b = {m2.params['disclaimer:framing']:+.4f}  "
        f"p = {m2.pvalues['disclaimer:framing']:.4f}")
    log(f"    H3  b = {m2.params['disclaimer:apology']:+.4f}  "
        f"p = {m2.pvalues['disclaimer:apology']:.4f}")

    rows = [{"model": "Proposed main (episode controls)",
             "H2_b": round(m_main.params["disclaimer:framing"], 4),
             "H2_p": round(m_main.pvalues["disclaimer:framing"], 4),
             "H3_b": round(m_main.params["disclaimer:apology"], 4),
             "H3_p": round(m_main.pvalues["disclaimer:apology"], 4),
             "NagelkerkeR2": round(nagelkerke(m_main), 4)}]
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "proposed_main_model.csv"),
                              index=False)

    coefs = pd.DataFrame({"B": m_main.params.round(4), "SE": m_main.bse.round(4),
                          "z": m_main.tvalues.round(3), "p": m_main.pvalues.round(4)})
    coefs["sig"] = coefs["p"].apply(stars)
    coefs.to_csv(os.path.join(args.outdir, "proposed_main_model_full.csv"))

    # =====================================================================
    section("Q2 (WASKO). SENTIMENT AS DEPENDENT VARIABLE")
    if not has_sent:
        log("  sentiment not available in this file - skipped")
    else:
        s = d.dropna(subset=["sentiment_after_refusal"]).copy()
        log(f"  N = {len(s):,}, mean = {s['sentiment_after_refusal'].mean():.4f}, "
            f"SD = {s['sentiment_after_refusal'].std():.4f}")

        r1 = smf.ols(f"sentiment_after_refusal ~ {BASE}", data=s).fit()
        show(r1, ["disclaimer", "framing", "apology", "disclaimer:framing",
                  "disclaimer:apology", "framing:apology"],
             "S1: main effects + interactions, no controls", logit=False)

        sc = s.dropna(subset=["english"]).copy()
        f2 = f"sentiment_after_refusal ~ {BASE} + turn + english + {T}"
        if has_tox:
            f2 += " + tox_c"
        r2 = smf.ols(f2, data=sc).fit()
        show(r2, ["disclaimer", "framing", "apology", "disclaimer:framing",
                  "disclaimer:apology", "framing:apology", "tox_c", "turn", "english"],
             "S2: same specification as the proposed main model", logit=False)

        r3 = smf.ols(f2, data=sc).fit(cov_type="cluster",
                                      cov_kwds={"groups": sc["model_grp"]})
        show(r3, ["disclaimer", "framing", "apology", "disclaimer:framing",
                  "disclaimer:apology", "framing:apology"],
             "S3: S2 with standard errors clustered by responding model", logit=False)

        out = pd.DataFrame({
            "term": r2.params.index, "B": r2.params.values.round(4),
            "SE": r2.bse.values.round(4), "t": r2.tvalues.values.round(3),
            "p": r2.pvalues.values.round(4)})
        out["sig"] = out["p"].apply(stars)
        out.to_csv(os.path.join(args.outdir, "sentiment_as_dv.csv"), index=False)

        log("\n  Are the interactions significant on sentiment?")
        for t, h in [("disclaimer:framing", "H2"), ("disclaimer:apology", "H3")]:
            v = "YES" if r2.pvalues[t] < .05 else "no"
            log(f"    {h}: {v}  (b = {r2.params[t]:+.4f}, p = {r2.pvalues[t]:.4f})")

    # =====================================================================
    section("Q3 (WASKO). IS FRAMING THE 'STRONGEST' PREDICTOR OF SENTIMENT?")
    log("Comparing raw b-values is not a test. The correct test is whether")
    log("two coefficients differ significantly from each other (Wald test).")
    if has_sent:
        s = d.dropna(subset=["sentiment_after_refusal"]).copy()
        r = smf.ols(f"sentiment_after_refusal ~ disclaimer + framing + apology",
                    data=s).fit()
        log(f"\n  main-effects-only model, N = {len(s):,}")
        for t in ["disclaimer", "framing", "apology"]:
            log(f"    {t:12s} b = {r.params[t]:+.4f}  SE = {r.bse[t]:.4f}  "
                f"p = {r.pvalues[t]:.4f} {stars(r.pvalues[t])}")
        log("\n  pairwise equality tests:")
        pairs = [("framing", "disclaimer"), ("framing", "apology"),
                 ("disclaimer", "apology")]
        res_rows = []
        for a, b in pairs:
            t = r.t_test(f"{a} - {b} = 0")
            pv = float(np.ravel(t.pvalue)[0])
            log(f"    {a} vs {b}: difference = {float(t.effect[0]):+.4f}, "
                f"p = {pv:.4f} {stars(pv)}")
            res_rows.append({"comparison": f"{a} vs {b}",
                             "difference": round(float(t.effect[0]), 4),
                             "p": round(pv, 4)})
        pd.DataFrame(res_rows).to_csv(
            os.path.join(args.outdir, "sentiment_coefficient_tests.csv"), index=False)
    else:
        log("  sentiment not available - skipped")

    # =====================================================================
    section("Q4 (WASKO). DROPPING THE 163 'UNCATEGORIZED' EPISODES")
    log("Wasko asked whether removing them moves the H2 p-value below .05.")
    log("Reporting both directions; see the note at the end of this section.")

    d["backlash_nouncat"] = d["user_response_category"].isin(
        [c for c in BACKLASH if c != "uncategorized"]).astype(int)
    dc["backlash_nouncat"] = dc["user_response_category"].isin(
        [c for c in BACKLASH if c != "uncategorized"]).astype(int)
    log(f"\n  backlash with uncategorized:    {int(d['backlash'].sum()):,} "
        f"({100 * d['backlash'].mean():.1f}%)")
    log(f"  backlash without uncategorized: {int(d['backlash_nouncat'].sum()):,} "
        f"({100 * d['backlash_nouncat'].mean():.1f}%)")

    comp = []
    specs = [("Model 2 (no controls)", f"backlash ~ {BASE}", d),
             ("Proposed main model", f_main, dc)]
    M = "C(model_grp, Treatment('vicuna-13b'))"
    f_full = f_main + f" + {M}"
    specs.append(("Full model (with model FE)", f_full, dc))

    for name, formula, data in specs:
        for dv, tag in [("backlash", "with uncat"), ("backlash_nouncat", "without uncat")]:
            try:
                rr = smf.logit(formula.replace("backlash ~", f"{dv} ~"),
                               data=data).fit(disp=0, maxiter=200)
                comp.append({
                    "spec": name, "outcome": tag,
                    "H2_b": round(rr.params["disclaimer:framing"], 4),
                    "H2_p": round(rr.pvalues["disclaimer:framing"], 4),
                    "H3_b": round(rr.params["disclaimer:apology"], 4),
                    "H3_p": round(rr.pvalues["disclaimer:apology"], 4)})
            except Exception as e:
                log(f"    {name} / {tag} failed: {e}")
    comp = pd.DataFrame(comp)
    log("")
    log(comp.to_string(index=False))
    comp.to_csv(os.path.join(args.outdir, "uncategorized_sensitivity.csv"), index=False)

    log("\n  NOTE. Choosing the outcome definition on the basis of whether the")
    log("  p-value crosses .05 is exactly what reviewers at top journals look")
    log("  for. The decision should be made on conceptual grounds and the")
    log("  result reported either way. The conceptual case for dropping the")
    log("  category is strong on its own: it is a parsing residual, not a")
    log("  behaviour. That case does not depend on which way the p-value moves.")

    # =====================================================================
    section("Q5 (SARAH). TOXICITY VERSUS SEVERITY")
    log("Sarah asks whether request toxicity and request severity are the same")
    log("thing. They are not, and the data can show it.")
    if has_tox:
        log("\n  toxicity by request topic:")
        by_t = d.groupby("topic").agg(
            n=("tox", "size"),
            mean_toxicity=("tox", lambda x: round(x.mean(), 4)),
            pct_above_50=("tox", lambda x: round(100 * (x > .5).mean(), 1)),
            backlash_pct=("backlash", lambda x: round(100 * x.mean(), 1)),
        ).sort_values("mean_toxicity", ascending=False)
        log(by_t.to_string())
        by_t.to_csv(os.path.join(args.outdir, "toxicity_by_topic.csv"))

        vv = (d["topic"] == "value_violation").astype(int)
        r_pb = stats.pointbiserialr(vv, d["tox"])
        log(f"\n  correlation between toxicity and value-violation requests:")
        log(f"    r = {r_pb.statistic:.4f}, p = {r_pb.pvalue:.3g}")
        log(f"    toxicity | value violation      = {d.loc[vv == 1, 'tox'].mean():.4f}")
        log(f"    toxicity | instrumental request = {d.loc[vv == 0, 'tox'].mean():.4f}")
        log("\n  A weak correlation means the two capture different things:")
        log("  toxicity is about how hostile the wording is, severity is about")
        log("  how harmful compliance would be. A politely worded request for")
        log("  something dangerous is low toxicity and high severity.")
    else:
        log("  toxicity not available - skipped")

    section("DONE")
    log(f"Written to {os.path.abspath(args.outdir)}")
    with open(os.path.join(args.outdir, "followup_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG))


if __name__ == "__main__":
    main()
