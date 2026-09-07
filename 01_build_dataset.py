#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 20251007
np.random.seed(SEED)

BACKLASH_CATEGORIES = [
    "challenges_refusal",
    "repeats_original_request",
    "clarification_request",
    "uncategorized",
]

NON_BACKLASH_CATEGORIES = [
    "topic_shift",
    "compliant_reformulation",
    "takes_alternative",
    "accepts_boundary",
    "abandonment",
]

FOCAL_FEATURES = [
    "has_ai_identity_disclaimer",
    "has_normative_value_signaling",
    "has_apology_regret",
]

PAPER_BENCHMARKS = {
    "n_confirmed": 12185,
    "n_backlash": 2351,
    "n_disclaimer": 4424,
    "n_framing": 3213,
    "n_apology": 8484,
    "n_lang_unknown": 512,
}


def log(msg=""):
    print(msg, flush=True)


def section(title):
    log("\n" + "=" * 74)
    log(title)
    log("=" * 74)


def to_bool_strict(series):
    raw = series.astype(str).str.strip()
    upper = raw.str.upper()
    is_true = upper == "TRUE"
    is_false = upper == "FALSE"
    malformed = ~(is_true | is_false)
    return is_true, malformed


def clean_text(s, max_chars=2000):
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    t = str(s).replace("\x00", " ").strip()
    return t[:max_chars]


def pick_device(requested):
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_cache(path):
    if os.path.exists(path):
        log(f"  [cache] loading {path}")
        return pd.read_pickle(path)
    return None


def save_cache(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_pickle(path)


def score_sentiment(texts, device, batch_size, cache_path, model_name):
    cached = load_cache(cache_path)
    if cached is not None:
        return cached

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    log(f"  loading {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.eval().to(device)

    id2label = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
    order = []
    for want in ("negative", "neutral", "positive"):
        hit = [i for i, lab in id2label.items() if want in lab or lab == want[:3]]
        order.append(hit[0] if hit else None)
    if any(o is None for o in order):
        order = [0, 1, 2]

    probs = np.zeros((len(texts), 3), dtype=np.float32)
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for bi in range(n_batches):
            lo = bi * batch_size
            hi = min(lo + batch_size, len(texts))
            batch = [t if t else "." for t in texts[lo:hi]]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to(device)
            p = torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()
            probs[lo:hi] = p[:, order]
            if bi % 50 == 0 or bi == n_batches - 1:
                log(f"    sentiment batch {bi + 1}/{n_batches}")

    out = pd.DataFrame({
        "sentiment_neg": probs[:, 0],
        "sentiment_neu": probs[:, 1],
        "sentiment_pos": probs[:, 2],
    })
    out["sentiment_after_refusal"] = out["sentiment_pos"] - out["sentiment_neg"]
    out["sentiment_label"] = np.array(
        ["negative", "neutral", "positive"])[probs.argmax(axis=1)]

    empty = np.array([len(t.strip()) == 0 for t in texts])
    out.loc[empty, :] = np.nan
    log(f"  {int(empty.sum())} rows had empty text and were left missing")

    save_cache(out, cache_path)
    return out


def score_toxicity(texts, device, batch_size, cache_path, model_name):
    cached = load_cache(cache_path)
    if cached is not None:
        return cached

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    log(f"  loading {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.eval().to(device)

    n_labels = mdl.config.num_labels
    id2label = getattr(mdl.config, "id2label", {}) or {}
    names = []
    for i in range(n_labels):
        raw = str(id2label.get(i, id2label.get(str(i), f"label_{i}")))
        names.append(raw.lower().replace(" ", "_").replace("-", "_"))
    log(f"  {n_labels} output labels: {names}")

    scores = np.zeros((len(texts), n_labels), dtype=np.float32)
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for bi in range(n_batches):
            lo = bi * batch_size
            hi = min(lo + batch_size, len(texts))
            batch = [t if t else "." for t in texts[lo:hi]]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            scores[lo:hi] = torch.sigmoid(logits).cpu().numpy()
            if bi % 50 == 0 or bi == n_batches - 1:
                log(f"    toxicity batch {bi + 1}/{n_batches}")

    out = pd.DataFrame(scores, columns=names)
    empty = np.array([len(t.strip()) == 0 for t in texts])
    out.loc[empty, :] = np.nan

    save_cache(out, cache_path)
    return out


def export_validation_samples(df, outdir, n_per_cell=60):
    vdir = os.path.join(outdir, "validation_samples")
    os.makedirs(vdir, exist_ok=True)
    rng = np.random.RandomState(SEED)

    def take(frame, n, cols, name):
        n = min(n, len(frame))
        if n == 0:
            return
        idx = rng.choice(frame.index.values, size=n, replace=False)
        sub = frame.loc[sorted(idx), cols].copy()
        sub.insert(0, "CHECK_ok_yes_no", "")
        sub.insert(1, "CHECK_comment", "")
        sub.to_csv(os.path.join(vdir, f"{name}.csv"),
                   index=False, encoding="utf-8-sig")
        log(f"  wrote {name}.csv  (n={len(sub)})")

    base = ["conv_id", "model", "language", "turn",
            "user_prompt", "assistant_refusal", "user_follow_up"]

    take(df[df["_is_refusal"]], n_per_cell,
         base + ["is_actual_refusal"], "01_is_actual_refusal_TRUE")
    take(df[(~df["_is_refusal"]) & (~df["_refusal_malformed"])], n_per_cell,
         base + ["is_actual_refusal"], "02_is_actual_refusal_FALSE")

    conf = df[df["_is_refusal"]]

    parts = []
    for cat, grp in conf.groupby("user_response_category"):
        k = min(25, len(grp))
        idx = rng.choice(grp.index.values, size=k, replace=False)
        parts.append(conf.loc[sorted(idx)])
    if parts:
        sub = pd.concat(parts).sort_values("user_response_category")
        sub = sub[base + ["user_response_category", "backlash"]].copy()
        sub.insert(0, "CHECK_ok_yes_no", "")
        sub.insert(1, "CHECK_comment", "")
        sub.to_csv(os.path.join(vdir, "03_user_response_category.csv"),
                   index=False, encoding="utf-8-sig")
        log(f"  wrote 03_user_response_category.csv  (n={len(sub)})")

    marker_files = {
        "has_apology_regret": "04_apology",
        "has_ai_identity_disclaimer": "05_ai_disclaimer",
        "has_normative_value_signaling": "06_normative_framing",
    }
    for col, stub in marker_files.items():
        take(conf[conf[col]], 40, base + [col], f"{stub}_TRUE")
        take(conf[~conf[col]], 40, base + [col], f"{stub}_FALSE")


def export_appendix_frequencies(df, outdir):
    conf = df[df["_is_refusal"]]
    rows = []

    for col in FOCAL_FEATURES:
        rows.append({"block": "refusal feature", "variable": col,
                     "level": "present", "n": int(conf[col].sum()),
                     "pct": round(100 * conf[col].mean(), 2)})
        rows.append({"block": "refusal feature", "variable": col,
                     "level": "absent", "n": int((~conf[col]).sum()),
                     "pct": round(100 * (~conf[col]).mean(), 2)})

    for cat in BACKLASH_CATEGORIES + NON_BACKLASH_CATEGORIES:
        n = int((conf["user_response_category"] == cat).sum())
        rows.append({"block": "backlash = 1" if cat in BACKLASH_CATEGORIES
                     else "backlash = 0",
                     "variable": "user_response_category", "level": cat,
                     "n": n, "pct": round(100 * n / len(conf), 2)})

    for cat, n in conf["prompt_category"].value_counts().items():
        rows.append({"block": "request topic", "variable": "prompt_category",
                     "level": cat, "n": int(n),
                     "pct": round(100 * n / len(conf), 2)})

    for cat, n in conf["model"].value_counts().items():
        rows.append({"block": "model", "variable": "model", "level": cat,
                     "n": int(n), "pct": round(100 * n / len(conf), 2)})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "appendix_b_frequencies.csv"),
               index=False, encoding="utf-8-sig")
    log(f"  wrote appendix_b_frequencies.csv  ({len(out)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--scope", choices=["all", "confirmed"], default="all")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sentiment-model",
                    default="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    ap.add_argument("--toxicity-model",
                    default="unitary/multilingual-toxic-xlm-roberta")
    ap.add_argument("--skip-sentiment", action="store_true")
    ap.add_argument("--skip-toxicity", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cache_dir = os.path.join(args.outdir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    device = pick_device(args.device)

    section("1. LOADING RAW DATA")
    df = pd.read_csv(args.input, low_memory=False)
    original_columns = list(df.columns)
    log(f"  {df.shape[0]:,} rows x {df.shape[1]} columns")
    if args.limit:
        df = df.head(args.limit).copy()
        log(f"  limit active: truncated to {len(df):,} rows")

    section("2. DATA INTEGRITY")
    report = []

    is_ref, malformed = to_bool_strict(df["is_actual_refusal"])
    df["_is_refusal"] = is_ref
    df["_refusal_malformed"] = malformed

    log(f"  is_actual_refusal TRUE      : {int(is_ref.sum()):,}")
    log(f"  is_actual_refusal FALSE     : {int((~is_ref & ~malformed).sum()):,}")
    log(f"  is_actual_refusal MALFORMED : {int(malformed.sum())}")
    report.append(f"Rows total: {len(df)}")
    report.append(f"Confirmed refusals (TRUE): {int(is_ref.sum())}")
    report.append(f"Malformed is_actual_refusal cells: {int(malformed.sum())}")

    if malformed.any():
        bad = df.loc[malformed, ["conv_id", "model", "is_actual_refusal"]]
        bad.to_csv(os.path.join(args.outdir, "malformed_rows.csv"),
                   index=False, encoding="utf-8-sig")
        log("  malformed rows written to malformed_rows.csv")
        log("  these are excluded from the analysis sample either way")
        for _, r in bad.iterrows():
            report.append(f"  MALFORMED conv_id={r['conv_id']} "
                          f"model={r['model']} "
                          f"value={repr(str(r['is_actual_refusal'])[:80])}")

    df["backlash"] = df["user_response_category"].isin(
        BACKLASH_CATEGORIES).astype(int)

    unknown_cats = set(df["user_response_category"].dropna().unique()) - \
        set(BACKLASH_CATEGORIES) - set(NON_BACKLASH_CATEGORIES)
    if unknown_cats:
        log(f"  unexpected response categories: {unknown_cats}")
        report.append(f"Unexpected response categories: {unknown_cats}")

    if not args.limit:
        conf = df[df["_is_refusal"]]
        checks = {
            "n_confirmed": len(conf),
            "n_backlash": int(conf["backlash"].sum()),
            "n_disclaimer": int(conf["has_ai_identity_disclaimer"].sum()),
            "n_framing": int(conf["has_normative_value_signaling"].sum()),
            "n_apology": int(conf["has_apology_regret"].sum()),
            "n_lang_unknown": int((conf["language"] == "unknown").sum()),
        }
        log("\n  Reproduction check against the manuscript:")
        all_ok = True
        for k, v in checks.items():
            exp = PAPER_BENCHMARKS[k]
            ok = (v == exp)
            all_ok &= ok
            log(f"    {'OK ' if ok else 'XX '} {k:16s} got {v:6,d}  paper {exp:6,d}")
            report.append(f"{k}: got {v}, paper {exp}, match={ok}")
        if all_ok:
            log("  --> all manuscript numbers reproduce exactly.")
        else:
            log("  --> MISMATCH. Stop and investigate before writing anything up.")

    if args.scope == "confirmed":
        mask = df["_is_refusal"].values
    else:
        mask = np.ones(len(df), dtype=bool)
    log(f"\n  scoring scope: {args.scope} ({int(mask.sum()):,} rows)")

    followup = [clean_text(t) for t in df["user_follow_up"].values]
    prompt = [clean_text(t) for t in df["user_prompt"].values]
    idx_scored = np.where(mask)[0]

    if not args.skip_sentiment:
        section("3. SENTIMENT ON THE USER FOLLOW-UP (WASKO)")
        sent = score_sentiment([followup[i] for i in idx_scored], device,
                               args.batch_size,
                               os.path.join(cache_dir, "sentiment.pkl"),
                               args.sentiment_model)
        for col in ["sentiment_after_refusal", "sentiment_label",
                    "sentiment_neg", "sentiment_neu", "sentiment_pos"]:
            df[col] = np.nan
            df.loc[df.index[idx_scored], col] = sent[col].values
        log(f"  filled sentiment_after_refusal for "
            f"{df['sentiment_after_refusal'].notna().sum():,} rows")
    else:
        log("\n  skipped sentiment")

    if not args.skip_toxicity:
        section("4. TOXICITY (JOHANNES)")
        log("  scoring the user prompt")
        tox_p = score_toxicity([prompt[i] for i in idx_scored], device,
                               args.batch_size,
                               os.path.join(cache_dir, "tox_prompt.pkl"),
                               args.toxicity_model)
        log("  scoring the user follow-up")
        tox_f = score_toxicity([followup[i] for i in idx_scored], device,
                               args.batch_size,
                               os.path.join(cache_dir, "tox_followup.pkl"),
                               args.toxicity_model)

        for src, tag in ((tox_p, "prompt"), (tox_f, "followup")):
            for c in src.columns:
                col = f"tox_{tag}_{c}"
                df[col] = np.nan
                df.loc[df.index[idx_scored], col] = src[c].values
            mx = src.max(axis=1, skipna=True)
            df[f"toxicity_{tag}_max"] = np.nan
            df.loc[df.index[idx_scored], f"toxicity_{tag}_max"] = mx.values

        df["toxicity_max"] = df[["toxicity_prompt_max",
                                 "toxicity_followup_max"]].max(axis=1)
        log(f"  toxicity_max filled for {df['toxicity_max'].notna().sum():,} rows")
        log(f"  toxicity_max mean = {df['toxicity_max'].mean():.4f}, "
            f"median = {df['toxicity_max'].median():.4f}")
    else:
        log("\n  skipped toxicity")

    section("5. CONTROL VARIABLES (JOHANNES)")

    df["english"] = (df["language"] == "English").astype(float)
    df.loc[df["language"] == "unknown", "english"] = np.nan
    log(f"  english: {int((df['english'] == 1).sum()):,} English, "
        f"{int((df['english'] == 0).sum()):,} non-English, "
        f"{int(df['english'].isna().sum()):,} unknown")

    df["topic"] = df["prompt_category"].astype(str)
    log(f"  topic: {df['topic'].nunique()} levels from prompt_category")
    log("  NOTE: prompt_category is deterministically linked to refusal_type")
    log("        (value_violation <-> value_based_refusal). Never enter both.")

    conf_counts = df.loc[df["_is_refusal"], "model"].value_counts()
    rare = set(conf_counts[conf_counts < 100].index)
    df["model_grp"] = df["model"].where(~df["model"].isin(rare), "other_small")
    log(f"  model_grp: {df['model_grp'].nunique()} levels "
        f"({len(rare)} models with n<100 collapsed into other_small)")

    df["log_turn"] = np.log1p(df["turn"])

    section("6. WRITING OUTPUT")

    new_cols = [c for c in df.columns
                if c not in original_columns and not c.startswith("_")]
    final_cols = original_columns + new_cols
    out = df[final_cols].copy()

    assert list(out.columns[:len(original_columns)]) == original_columns, \
        "original column order was altered"

    path = os.path.join(args.outdir, "Wasko_Final_Dataset_ENRICHED.csv")
    out.to_csv(path, index=False, encoding="utf-8")
    log(f"  wrote {path}")
    log(f"  {out.shape[0]:,} rows x {out.shape[1]} columns "
        f"({len(original_columns)} original + {len(new_cols)} new)")
    log(f"  new columns: {', '.join(new_cols)}")

    export_appendix_frequencies(df, args.outdir)
    export_validation_samples(df, args.outdir)

    report.append(f"\nNew columns appended: {new_cols}")
    report.append(f"Sentiment model: {args.sentiment_model}")
    report.append(f"Toxicity model: {args.toxicity_model}")
    report.append(f"Scope: {args.scope}; device: {device}; seed: {SEED}")
    with open(os.path.join(args.outdir, "data_integrity_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(report))
    log("  wrote data_integrity_report.txt")

    section("DONE")
    log("data_integrity_report.txt, appendix_b_frequencies.csv,")
    log("console output ready. Then run 02_analysis.py.")


if __name__ == "__main__":
    main()
