#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

BASE = "/content/drive/MyDrive/Wasko_JPIM"
OUT = f"{BASE}/data"

CH = "challenges_refusal"
RE = "repeats_original_request"
CR = "compliant_reformulation"
TS = "topic_shift"
TA = "takes_alternative"
AB = "accepts_boundary"
AD = "abandonment"
CL = "clarification_request"
XX = "MODERATION_ARTIFACT"

LABELS = [
    RE, TS, CH, CH, RE, CH, CH, TS, CH, TS,
    CH, CH, CH, CR, CH, RE, CH, CH, CH, XX,
    CH, CH, CH, CH, CH, RE, CH, RE, RE, CH,
    TS, TS, CH, TS, CH, CH, CH, RE, CH, CH,
    CH, RE, TS, CH, CH, XX, CH, CH, CH, TS,
    CH, CH, CH, CH, CH, CH, RE, CH, RE, XX,
    CH, CR, CH, CH, RE, TS, XX, CH, RE, CH,
    CH, XX, CH, CH, RE, CR, CH, CH, CH, RE,
    RE, CH, XX, CH, RE, CH, CR, XX, CH, CH,
    RE, CH, CH, CR, RE, TA, CH, CH, RE, CH,
    RE, XX, CH, CH, XX, RE, XX, CH, CH, RE,
    RE, RE, CH, TS, RE, XX, XX, RE, CH, CH,
    CH, RE, RE, CH, CH, TS, CH, CH, CR, XX,
    TS, RE, CH, CH, CH, XX, RE, RE, CH, CH,
    RE, CR, CH, TS, CH, CH, RE, CR, RE, CH,
    CH, TS, TS, RE, CH, RE, CR, RE, CH, CH,
    RE, RE, CH,
]

BACKLASH = ["challenges_refusal", "repeats_original_request",
            "clarification_request", "uncategorized"]

ARTIFACT_PATTERN = "VIOLATES OUR CONTENT MODERATION"


def main():
    os.makedirs(OUT, exist_ok=True)
    src = f"{BASE}/data/Wasko_Final_Dataset_ENRICHED.csv"
    df = pd.read_csv(src, low_memory=False)
    print(f"loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    conf = df["is_actual_refusal"].astype(str).str.strip().str.upper() == "TRUE"
    target_idx = df[conf & (df["user_response_category"] == "uncategorized")].index
    print(f"uncategorized episodes: {len(target_idx)}")

    if len(target_idx) != len(LABELS):
        raise SystemExit(f"label count {len(LABELS)} != episode count {len(target_idx)}")

    df["user_response_category_v1"] = df["user_response_category"]
    df["recoded"] = False
    df["coding_source"] = "llm_original"

    assigned = pd.Series(LABELS, index=target_idx)
    real = assigned[assigned != XX]
    df.loc[real.index, "user_response_category"] = real.values
    df.loc[target_idx, "recoded"] = True
    df.loc[real.index, "coding_source"] = "manual_authors"

    print("\nhand-coded label distribution (excluding artifacts):")
    for k, v in real.value_counts().items():
        print(f"  {k:26s} {v:4d}  ({100 * v / len(real):.1f}%)")

    n_bl = real.isin(BACKLASH).sum()
    print(f"\n{n_bl} of {len(real)} recoded episodes are backlash "
          f"({100 * n_bl / len(real):.1f}%); all {len(target_idx)} counted as "
          f"backlash before")

    fu = df["user_follow_up"].astype(str)
    artifact = fu.str.contains(ARTIFACT_PATTERN, case=False, na=False)
    df["moderation_artifact"] = artifact
    print(f"\nplatform moderation messages recorded as user turns: "
          f"{int((artifact & conf).sum())}")
    df.loc[artifact, "is_actual_refusal"] = "EXCLUDED_ARTIFACT"
    df.loc[artifact, "user_response_category"] = XX

    new_conf = df["is_actual_refusal"].astype(str).str.strip().str.upper() == "TRUE"
    c = df[new_conf]
    print(f"\nanalysis sample: {len(c):,} (was 12,185)")
    print(f"backlash: {int(c['user_response_category'].isin(BACKLASH).sum()):,} "
          f"({100 * c['user_response_category'].isin(BACKLASH).mean():.1f}%)")

    audit = df.loc[target_idx, [
        "conv_id", "model", "language", "user_prompt", "assistant_refusal",
        "user_follow_up", "user_response_category_v1"]].copy()
    audit["manual_label"] = assigned.values
    audit["is_artifact"] = (assigned == XX).values
    audit.to_csv(os.path.join(OUT, "manual_coding_163.csv"),
                 index=False, encoding="utf-8-sig")
    print("\nwrote manual_coding_163.csv")

    out = os.path.join(OUT, "Wasko_Final_Dataset_RECODED.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out}")
    print(f"{df.shape[0]:,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
