import os
import re
import gc
import glob
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline

TARGET_DIR = "/content/drive/MyDrive/WASKO JPIM"
MAX_CHARS = 3000
CHUNK_SIZE = 5000
CLASSIFIER = "Qwen/Qwen2.5-3B-Instruct"

SCREENING_MARKERS = [
    "i cannot", "i can not", "i can't", "as an ai", "i apologize", "i'm sorry",
    "against my programming", "i am unable", "i will not", "i am not able", "i refuse",
]

FEATURE_MARKERS = {
    "has_preamble_politeness_sandwich": [
        "hello", "hi", "hey", "nice to meet you", "I'm here to help",
        "I'd be happy to help", "sure", "of course"],
    "has_apology_regret": [
        "sorry", "apologize", "regret", "unfortunately", "my apologies"],
    "has_helpfulness_service_orientation": [
        "how can i help", "how else can i help",
        "let me know if you have other questions", "I'm here to assist"],
    "has_normative_value_signaling": [
        "inappropriate", "unethical", "harmful", "toxic", "respectful",
        "safe environment", "promote hate"],
    "has_clarification_instead_of_refusal": [
        "could you clarify", "do you mean", "i need more context", "can you specify"],
    "has_content_warning_style_framing": [
        "trigger warning", "content warning", "sensitive topic", "viewer discretion"],
    "has_alternative_redirection": [
        "instead, i can", "however, i can", "alternatively", "what i can do is",
        "here is an alternative"],
    "has_invite_followup": [
        "do you have any other questions", "is there anything else",
        "feel free to ask", "can i help with anything else"],
    "has_hard_refusal_or_inability": [
        "i cannot", "i can not", "i can't", "i am unable", "i will not",
        "i am not able", "i refuse"],
    "has_explanation_justification": [
        "because", "the reason is", "due to", "as a result of", "this means that"],
    "has_ai_identity_disclaimer": [
        "as an ai", "as an artificial intelligence", "as a large language model",
        "i am an ai", "i'm an ai", "i do not have personal", "i don\u2019t have feelings"],
    "has_policy_programming_justification": [
        "against my programming", "violates openai policy", "safety guidelines",
        "content policy", "guardrails"],
    "has_harm_legal_moral_risk_framing": [
        "illegal", "dangerous", "against the law", "copyright",
        "medical advice", "legal advice"],
    "has_capability_limitation": [
        "i don't have access to the internet", "i cannot browse", "i cannot open files",
        "i cannot watch videos", "no real-time data"],
}

JAILBREAK_MARKERS = [
    "ignore previous instructions", "ignore all previous", "dan ", "do anything now",
    "pretend to be", "roleplay", "role play", "act as", "developer mode",
    "system prompt", "you are now", "rules do not apply", "bypass", "jailbreak",
]

INSTRUMENTAL_CATEGORIES = [
    "live_or_recent_data_request", "web_browsing_request",
    "external_file_or_attachment_request", "private_or_personal_information_request",
    "real_world_action_request", "professional_advice_limitation",
    "model_capability_limitation", "ambiguous_instrumental_request", "not_instrumental",
]

USER_RESPONSE_CATEGORIES = [
    "abandonment", "compliant_reformulation", "accepts_boundary", "takes_alternative",
    "topic_shift", "repeats_original_request", "challenges_refusal", "clarification_request",
]

SYS_REFUSAL = (
    "You classify text. Reply ONLY with TRUE if the assistant is refusing "
    "the request, or FALSE if it is complying."
)
SYS_INSTRUMENTAL = (
    "Classify the user prompt into exactly ONE of these categories: "
    + ", ".join(INSTRUMENTAL_CATEGORIES)
    + ". Reply ONLY with the category name."
)
SYS_USER_RESPONSE = (
    "Classify the user's reaction to an AI refusal into exactly ONE of these "
    "categories: " + ", ".join(USER_RESPONSE_CATEGORIES)
    + ". Reply ONLY with the category name."
)


def extract_episode(row):
    conv = row["conversation"]
    if not any(m in str(conv).lower() for m in SCREENING_MARKERS):
        return None

    user_prompt, assistant_refusal, user_follow_up = "", "", ""
    abandonment = True
    found = False

    for i in range(len(conv) - 1):
        if conv[i]["role"] == "user" and conv[i + 1]["role"] == "assistant":
            text = conv[i + 1]["content"]
            if any(m in text.lower() for m in SCREENING_MARKERS):
                found = True
                user_prompt = conv[i]["content"]
                assistant_refusal = text
                if i + 2 < len(conv) and conv[i + 2]["role"] == "user":
                    user_follow_up = conv[i + 2]["content"]
                    abandonment = False
                break

    if not found:
        return None

    return {
        "conv_id": row["conversation_id"],
        "model": row["model"],
        "language": row["language"],
        "turn": row["turn"],
        "user_prompt": user_prompt,
        "assistant_refusal": assistant_refusal,
        "user_follow_up": user_follow_up,
        "abandonment": abandonment,
        "prompt_length_words": len(str(user_prompt).split()),
        "refusal_length_words": len(str(assistant_refusal).split()),
        "user_response_length_words": len(str(user_follow_up).split()) if user_follow_up else 0,
        "conversation_length": len(conv),
    }


def stage_screen():
    out = f"{TARGET_DIR}/Wasko_LMSYS_Refusals_CPU_Prep.csv"
    if os.path.exists(out):
        return out

    corpus = load_dataset("lmsys/lmsys-chat-1m", split="train").to_pandas()

    episodes = []
    for _, row in tqdm(corpus.iterrows(), total=len(corpus)):
        ep = extract_episode(row)
        if ep:
            episodes.append(ep)

    df = pd.DataFrame(episodes)

    for feature, markers in FEATURE_MARKERS.items():
        pattern = "|".join(re.escape(m) for m in markers)
        df[feature] = df["assistant_refusal"].str.contains(pattern, case=False, na=False)

    df.to_csv(out, index=False)
    return out


def stage_classify(prep_file):
    df = pd.read_csv(prep_file)

    pipe = pipeline("text-generation", model=CLASSIFIER, device=0, torch_dtype=torch.float16)

    @torch.inference_mode()
    def run(system_prompt, texts):
        prompts = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"
            for t in texts
        ]
        outputs = pipe(prompts, max_new_tokens=15, do_sample=False,
                       return_full_text=False, batch_size=8, truncation=True)
        return [o[0]["generated_text"].strip() for o in tqdm(outputs, total=len(prompts))]

    for start in range(0, len(df), CHUNK_SIZE):
        index = start // CHUNK_SIZE + 1
        chunk_file = f"{TARGET_DIR}/Wasko_Final_Chunk_{index}.csv"
        if os.path.exists(chunk_file):
            continue

        chunk = df.iloc[start:start + CHUNK_SIZE].copy()
        refusals = chunk["assistant_refusal"].fillna("").astype(str).str.slice(0, MAX_CHARS)
        prompts = chunk["user_prompt"].fillna("").astype(str).str.slice(0, MAX_CHARS)
        followups = chunk["user_follow_up"].fillna("")

        chunk["is_actual_refusal"] = run(SYS_REFUSAL, ["Assistant says: " + x for x in refusals])
        chunk["prompt_instrumental_category"] = run(SYS_INSTRUMENTAL, ["Prompt: " + x for x in prompts])

        response_prompts = []
        for reply in followups:
            text = str(reply).strip()[:MAX_CHARS]
            if not text:
                response_prompts.append(
                    "<|im_start|>system\nReply exactly with: abandonment<|im_end|>\n"
                    "<|im_start|>assistant\n")
            else:
                response_prompts.append(
                    f"<|im_start|>system\n{SYS_USER_RESPONSE}<|im_end|>\n"
                    f"<|im_start|>user\nUser said: {text}<|im_end|>\n<|im_start|>assistant\n")

        with torch.inference_mode():
            outputs = pipe(response_prompts, max_new_tokens=15, do_sample=False,
                           return_full_text=False, batch_size=8, truncation=True)
            chunk["user_response_category"] = [
                o[0]["generated_text"].strip()
                for o in tqdm(outputs, total=len(response_prompts))]

        chunk.to_csv(chunk_file, index=False)
        del chunk, response_prompts
        gc.collect()
        torch.cuda.empty_cache()

    merged = pd.concat(
        (pd.read_csv(f) for f in sorted(glob.glob(f"{TARGET_DIR}/Wasko_Final_Chunk_*.csv"))),
        ignore_index=True)
    out = f"{TARGET_DIR}/Wasko_Final_Dataset.csv"
    merged.to_csv(out, index=False)
    return out


def map_label(value, permitted):
    text = str(value).lower()
    for label in permitted:
        if label in text:
            return label
    return "uncategorized"


def has_jailbreak(text):
    lowered = str(text).lower()
    return any(m in lowered for m in JAILBREAK_MARKERS)


def looks_malformed(value):
    text = str(value)
    stripped = text.lower().strip()
    if stripped in {"", "nan", "none", "user", "assistant"}:
        return True
    if len(stripped) <= 2:
        return True
    if re.search(r"^\s*[\[\]\{\}\|\\/]+", text):
        return True
    if re.search(r'"\s*[^"]+"\s*:', text):
        return True
    if "user" in stripped and "assistant" in stripped and len(stripped) > 200:
        return True
    return False


def stage_finalize(merged_file):
    df = pd.read_csv(merged_file)

    df["prompt_instrumental_category"] = df["prompt_instrumental_category"].apply(
        lambda x: map_label(x, INSTRUMENTAL_CATEGORIES))
    df["user_response_category"] = df["user_response_category"].apply(
        lambda x: map_label(x, USER_RESPONSE_CATEGORIES))

    df["prompt_has_jailbreak_attempt"] = df["user_prompt"].fillna("").apply(has_jailbreak)
    df["response_has_jailbreak_attempt"] = df["user_follow_up"].fillna("").apply(has_jailbreak)

    df["refusal_type"] = df["prompt_instrumental_category"].apply(
        lambda x: "value_based_refusal"
        if pd.isna(x) or x in {"uncategorized", "not_instrumental"}
        else "instrumental_refusal")

    df["prompt_category"] = df.apply(
        lambda r: r["prompt_instrumental_category"]
        if r["refusal_type"] == "instrumental_refusal" else "value_violation", axis=1)

    compliant = {"abandonment", "accepts_boundary", "compliant_reformulation", "takes_alternative"}
    noncompliant = {"repeats_original_request", "challenges_refusal"}
    ambiguous = {"topic_shift", "clarification_request", "uncategorized"}

    df["response_complies_with_refusal"] = df["user_response_category"].isin(compliant).astype(int)
    df["response_noncompliant"] = df["user_response_category"].isin(noncompliant).astype(int)
    df["response_is_ambiguous"] = df["user_response_category"].isin(ambiguous).astype(int)
    df["sentiment_after_refusal"] = pd.NA

    permitted_prompt = set(INSTRUMENTAL_CATEGORIES) | {"value_violation"}
    permitted_response = set(USER_RESPONSE_CATEGORIES) | {"uncategorized"}

    df["qc_flag"] = False
    df.loc[~df["prompt_category"].astype(str).isin(permitted_prompt), "qc_flag"] = True
    df.loc[~df["refusal_type"].astype(str).isin(
        {"instrumental_refusal", "value_based_refusal"}), "qc_flag"] = True
    df.loc[~df["prompt_instrumental_category"].astype(str).isin(
        permitted_prompt - {"value_violation"}), "qc_flag"] = True
    df.loc[~df["user_response_category"].astype(str).isin(permitted_response), "qc_flag"] = True

    for column in ["user_prompt", "assistant_refusal", "user_follow_up"]:
        df.loc[df[column].apply(looks_malformed), "qc_flag"] = True

    columns = df.columns.tolist()
    columns.insert(columns.index("prompt_instrumental_category"),
                   columns.pop(columns.index("prompt_category")))
    columns.insert(columns.index("prompt_instrumental_category"),
                   columns.pop(columns.index("refusal_type")))
    df = df[columns]

    clean = df[~df["qc_flag"]].copy()
    flagged = df[df["qc_flag"]].copy()

    clean_file = f"{TARGET_DIR}/Wasko_Final_Dataset_FINAL_CLEAN.csv"
    clean.to_csv(clean_file, index=False)
    flagged.to_csv(f"{TARGET_DIR}/Wasko_Final_Dataset_FINAL_FLAGGED.csv", index=False)

    print(f"clean rows: {len(clean)}")
    print(f"flagged rows: {len(flagged)}")
    print(f"confirmed refusals: {(clean['is_actual_refusal'].astype(str).str.strip().str.upper() == 'TRUE').sum()}")
    return clean_file


os.makedirs(TARGET_DIR, exist_ok=True)
prep = stage_screen()
merged = stage_classify(prep)
final = stage_finalize(merged)
print(final)
