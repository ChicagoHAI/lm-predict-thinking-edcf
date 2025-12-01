import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
RAW_DIR = RESULTS_DIR / "raw"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)
RAW_DIR.mkdir(exist_ok=True, parents=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EVAL_SIZE = 20
SEED = 42


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ExampleResult:
    idx: int
    question: str
    gold_answer: str
    condition: str
    predicted_tokens: Optional[int]
    reasoning_tokens: int
    total_new_tokens: int
    prediction_error: Optional[int]
    final_answer: str
    is_correct: bool


def load_gsm8k(path: Path, n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    data = []
    with path.open() as f:
        for line in f:
            data.append(json.loads(line))
    return rng.sample(data, n)


def extract_gold_answer(answer: str) -> str:
    match = re.search(r"####\\s*([\\-\\d\\.]+)", answer)
    return match.group(1).strip() if match else answer.strip()


def extract_final_numeric(text: str) -> str:
    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", text)
    return numbers[-1] if numbers else text.strip()


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_prompts(question: str) -> Dict[str, str]:
    cot_prompt = (
        "You are a math tutor. Solve the problem step by step. "
        "Provide a short reasoning labeled 'Reasoning:' followed by 'Final Answer:' "
        "with only the numeric answer.\n\n"
        f"Problem: {question}\n"
        "Reasoning:"
    )
    predict_prompt = (
        "You are a math tutor. First, estimate how many reasoning tokens you will need "
        "before giving the final answer. Respond using exactly this format:\n"
        "Predicted tokens: <integer>\nReasoning: <concise steps>\nFinal Answer: <numeric only>\n"
        "Keep reasoning under 200 tokens and keep 'Predicted tokens' as an integer.\n\n"
        f"Problem: {question}\n"
        "Predicted tokens:"
    )
    return {"cot": cot_prompt, "predict": predict_prompt}


def parse_predict_response(text: str) -> Dict[str, Optional[str]]:
    token_match = re.search(r"Predicted tokens[:\s]*(-?\d+)", text, re.I)
    reasoning_match = re.search(r"Reasoning:(.*?)(Final Answer:.*)", text, re.S | re.I)
    final_match = re.search(r"Final Answer[:\s]*([^\-\n\r]+)", text, re.I)

    predicted_tokens = int(token_match.group(1)) if token_match else None
    if predicted_tokens is None:
        # Sometimes the model directly emits the integer at the start.
        leading_int = re.match(r"\s*(-?\d+)", text)
        if leading_int:
            predicted_tokens = int(leading_int.group(1))

    reasoning = ""
    final_answer = text
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        final_answer = reasoning_match.group(2)
    elif final_match:
        final_answer = final_match.group(1)

    return {
        "predicted_tokens": predicted_tokens,
        "reasoning": reasoning if reasoning else text,
        "final_answer": final_answer,
    }


def parse_cot_response(text: str) -> Dict[str, str]:
    reasoning = ""
    final_answer = text
    match = re.search(r"Reasoning:(.*?)(Final Answer:.*)", text, re.S | re.I)
    if match:
        reasoning = match.group(1).strip()
        final_answer = match.group(2)
    return {"reasoning": reasoning, "final_answer": final_answer}


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_response(model, tokenizer, device, prompt: str, max_new_tokens: int = 192) -> Dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {"text": text, "new_tokens": len(generated_ids)}


def evaluate_subset(dataset_path: Path, subset_size: int = EVAL_SIZE, seed: int = SEED) -> pd.DataFrame:
    set_seed(seed)
    model, tokenizer, device = load_model()
    data = load_gsm8k(dataset_path, subset_size, seed)

    results: List[ExampleResult] = []

    for idx, example in enumerate(tqdm(data, desc="Running examples")):
        gold = extract_gold_answer(example["answer"])
        prompts = build_prompts(example["question"])

        # Baseline CoT
        cot_out = generate_response(model, tokenizer, device, prompts["cot"])
        cot_parsed = parse_cot_response(cot_out["text"])
        cot_reasoning = cot_parsed["reasoning"] or cot_out["text"]
        cot_answer = extract_final_numeric(cot_parsed["final_answer"])
        cot_tokens = count_tokens(tokenizer, cot_reasoning)

        results.append(
            ExampleResult(
                idx=idx,
                question=example["question"],
                gold_answer=gold,
                condition="cot",
                predicted_tokens=None,
                reasoning_tokens=cot_tokens,
                total_new_tokens=cot_out["new_tokens"],
                prediction_error=None,
                final_answer=cot_answer,
                is_correct=(cot_answer == gold),
            )
        )

        # Prediction-aware CoT
        pred_out = generate_response(model, tokenizer, device, prompts["predict"])
        pred_parsed = parse_predict_response(pred_out["text"])
        reasoning = pred_parsed["reasoning"] or pred_out["text"]
        final_ans = extract_final_numeric(pred_parsed["final_answer"])
        reasoning_tokens = count_tokens(tokenizer, reasoning)
        predicted_tokens = None
        if isinstance(pred_parsed["predicted_tokens"], int):
            predicted_tokens = pred_parsed["predicted_tokens"]
        else:
            # try to coerce if string
            try:
                predicted_tokens = int(str(pred_parsed["predicted_tokens"]))
            except (TypeError, ValueError):
                predicted_tokens = None

        prediction_error = (
            abs(predicted_tokens - reasoning_tokens) if predicted_tokens is not None else None
        )

        results.append(
            ExampleResult(
                idx=idx,
                question=example["question"],
                gold_answer=gold,
                condition="predict",
                predicted_tokens=predicted_tokens,
                reasoning_tokens=reasoning_tokens,
                total_new_tokens=pred_out["new_tokens"],
                prediction_error=prediction_error,
                final_answer=final_ans,
                is_correct=(final_ans == gold),
            )
        )

    df = pd.DataFrame([asdict(r) for r in results])
    return df


def compute_metrics(df: pd.DataFrame) -> Dict:
    metrics = {}
    for cond in df["condition"].unique():
        subset = df[df["condition"] == cond]
        metrics[cond] = {
            "n": len(subset),
            "accuracy": subset["is_correct"].mean(),
            "mean_reasoning_tokens": subset["reasoning_tokens"].mean(),
            "median_reasoning_tokens": subset["reasoning_tokens"].median(),
            "total_tokens": subset["total_new_tokens"].sum(),
        }
        if cond == "predict":
            pred_subset = subset.dropna(subset=["prediction_error"])
            if len(pred_subset) > 0:
                metrics[cond]["mae_predicted_tokens"] = pred_subset["prediction_error"].mean()
                metrics[cond]["bias_predicted_tokens"] = (
                    pred_subset["predicted_tokens"] - pred_subset["reasoning_tokens"]
                ).mean()
                corr = (
                    pred_subset[["predicted_tokens", "reasoning_tokens"]]
                    .corr()
                    .iloc[0, 1]
                )
                metrics[cond]["corr_pred_vs_actual"] = None if pd.isna(corr) else corr
    # Accuracy per 1k reasoning tokens
    for cond in df["condition"].unique():
        subset = df[df["condition"] == cond]
        tok_total = subset["reasoning_tokens"].sum()
        metrics[cond]["accuracy_per_1k_tokens"] = (
            metrics[cond]["accuracy"] / (tok_total / 1000) if tok_total > 0 else 0
        )
    return metrics


def save_outputs(df: pd.DataFrame, metrics: Dict) -> None:
    raw_path = RAW_DIR / "token_prediction_runs.parquet"
    metrics_path = RESULTS_DIR / "metrics.json"
    df.to_parquet(raw_path, index=False)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return str(obj)


def plot_results(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    predict_df = df[df["condition"] == "predict"].dropna(subset=["predicted_tokens"])
    if not predict_df.empty:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=predict_df,
            x="predicted_tokens",
            y="reasoning_tokens",
            hue="is_correct",
            palette={True: "tab:green", False: "tab:red"},
        )
        plt.plot([0, predict_df["reasoning_tokens"].max()], [0, predict_df["reasoning_tokens"].max()], "k--")
        plt.xlabel("Predicted reasoning tokens")
        plt.ylabel("Actual reasoning tokens")
        plt.title("Prediction vs. Actual Tokens")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "pred_vs_actual.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.histplot(predict_df["prediction_error"], bins=20, kde=True)
        plt.xlabel("Absolute prediction error (tokens)")
        plt.title("Prediction Error Distribution")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "prediction_error_hist.png", dpi=200)
        plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="condition", y="reasoning_tokens", errorbar="sd")
    plt.ylabel("Reasoning tokens (mean ± sd)")
    plt.title("Token Cost by Condition")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "token_cost_by_condition.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    acc_df = (
        df.groupby("condition")["is_correct"].mean().reset_index().rename(columns={"is_correct": "accuracy"})
    )
    sns.barplot(data=acc_df, x="condition", y="accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy by Condition")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_by_condition.png", dpi=200)
    plt.close()


def main():
    dataset_path = Path("datasets/gsm8k_test.jsonl")
    df = evaluate_subset(dataset_path, subset_size=EVAL_SIZE, seed=SEED)
    metrics = compute_metrics(df)
    save_outputs(df, metrics)
    plot_results(df)
    print(json.dumps(metrics, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
