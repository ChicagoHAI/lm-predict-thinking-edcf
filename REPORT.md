## 1. Executive Summary
- Research question: Can language models predict how many reasoning tokens they will use (“thinking time”) before responding?
- Key finding: On a 20-example GSM8K subset using a small local Qwen2.5-0.5B-Instruct model, predicted token counts were weakly correlated with actual reasoning tokens (ρ≈0.14) with large MAE (~198 tokens) and zero answer accuracy in both baseline and prediction-aware settings.
- Practical implication: Lightweight local models do not reliably introspect their required reasoning budget; stronger models or API-grade systems are likely needed for meaningful self-budgeting.

## 2. Goal
- Hypothesis tested: LMs can predict their reasoning-token budget and use it to improve efficiency without hurting accuracy.
- Importance: Accurate token budgeting could reduce latency/cost and provide user-facing ETA for long reasoning traces.
- Expected impact: If feasible, enables budget-aware decoding and early-exit policies; current evidence (with a small model) does not support the hypothesis.

## 3. Data Construction
### Dataset Description
- Source: GSM8K (grade-school math reasoning). Files: `datasets/gsm8k_train.jsonl`, `datasets/gsm8k_test.jsonl`.
- Subset: 20 examples sampled from `gsm8k_test.jsonl`.
- Characteristics: Short word problems with numeric answers, CoT-friendly.
- Known biases/limits: Arithmetic-focused, English-only, single final numeric answer.

### Example Samples
| # | Question | Gold Answer |
|---|----------|-------------|
| 1 | “Natalia sold clips to 48 of her friends... How many clips did Natalia sell altogether in April and May?” | 72 |
| 2 | “Kim raises $320 more than Alexandra, who raises $430... How much money did they all raise?” | 1880 |

### Data Quality
- Missing values: none in subset.
- Duplicates: none.
- Outliers: not applicable (short text Q/A).
- Validation: Parsed gold answers from trailing `####` format.

### Preprocessing Steps
1. Loaded JSONL and randomly sampled 20 test examples (seed=42).
2. Extracted gold numeric answer via regex on `####`.
3. No additional cleaning; prompts consume raw question text.

### Train/Val/Test Splits
- Only test subset used (20 examples) for exploratory analysis; no training/fine-tuning performed.

## 4. Experiment Description
### Methodology
#### High-Level Approach
- Compare a baseline Chain-of-Thought (CoT) prompt vs. a prediction-aware prompt where the model first outputs an estimated reasoning-token count (`Predicted tokens: <int>`) before providing reasoning and the final answer.
- Use tokenizer-based counts of generated reasoning tokens as the “actual thinking tokens”.

#### Why This Method?
- Directly measures introspective budgeting without external regressors; keeps variables limited to prompt differences.
- Token counting via tokenizer mirrors model’s own tokenization.

#### Implementation Details
- Tools/Libraries: `transformers 4.57.3`, `torch 2.9.1`, `pandas 2.3.3`, `tiktoken 0.12.0` (token accounting), `matplotlib 3.10.7`, `seaborn 0.13.2`.
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (CPU inference).
- Sampling: deterministic (`temperature=0`, `max_new_tokens=192`), seed=42.

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| model_name | Qwen/Qwen2.5-0.5B-Instruct | small, CPU-friendly open-source |
| temperature | 0.0 | enforce determinism |
| max_new_tokens | 192 | cap reasoning length |
| subset_size | 20 | runtime/cost bound |

#### Training Procedure / Analysis Pipeline
1. Build prompts (baseline CoT, prediction-aware format).
2. Generate responses per example for both conditions.
3. Parse predicted tokens (integer at start or after label), reasoning text, and final numeric answer.
4. Count reasoning tokens via tokenizer; compute accuracy, MAE, bias, correlation.
5. Save runs to `results/raw/token_prediction_runs.parquet`, metrics to `results/metrics.json`, plots to `results/plots/`.

### Experimental Protocol
- Reproducibility: single deterministic run, seed=42; CPU-only.
- Hardware: CPU (no GPU), ~3 minutes runtime.
- Metrics: EM accuracy, reasoning-token counts, MAE/bias between predicted vs. actual tokens, token efficiency (accuracy per 1k reasoning tokens), correlation.

### Raw Results
| Condition | n | Accuracy | Mean Reasoning Tokens | Total New Tokens | MAE (pred vs actual) | Bias (pred-actual) | Corr (pred, actual) |
|-----------|---|----------|-----------------------|------------------|----------------------|--------------------|---------------------|
| CoT | 20 | 0.00 | 145.85 | 2933 | — | — | — |
| Predict-aware | 20 | 0.00 | 112.80 | 3786 | 198.22 | +7.33 | 0.136 |

Visualizations saved to `results/plots/`: `pred_vs_actual.png`, `prediction_error_hist.png`, `token_cost_by_condition.png`, `accuracy_by_condition.png`.

## 5. Result Analysis
### Key Findings
1. **No task accuracy**: The small model failed to answer any GSM8K examples correctly in either condition, limiting downstream efficiency conclusions.
2. **Weak budget prediction**: For 18/20 cases with a parsed prediction, MAE≈198 tokens and low correlation (ρ≈0.14); 83% of cases overran the predicted budget.
3. **Token cost difference**: Prediction-aware prompt shortened mean reasoning tokens (113 vs. 146) but increased total tokens because of format overhead; accuracy remained 0.

### Hypothesis Testing
- H1 (accurate self-token prediction): Not supported in this setting—low correlation and large errors.
- H2 (efficiency gains without accuracy loss): Not supported; no accuracy and modest token savings offset by format overhead.

### Surprises and Insights
- Model often emitted the predicted integer but appended it after the final answer, requiring lenient parsing.
- Predictions were sometimes extreme (e.g., 1880) showing poor calibration, likely due to weak task competence.

### Error Analysis
- Failure modes: arithmetic mistakes in reasoning, hallucinated large token predictions, inconsistent formatting (prediction sometimes appended after answer).
- Overrun rate: ~83% (actual reasoning tokens > predicted).

### Limitations
- Model capacity: 0.5B parameter model lacks reasoning skill; results unlikely to generalize to stronger LLMs.
- Sample size: 20 examples—adequate for a quick probe but not for strong statistical claims.
- Offline inference only; no API-grade models due to available keys, so hypothesis remains untested on capable systems.

## 6. Conclusions
- Small local Qwen2.5-0.5B-Instruct could not accurately predict or adhere to its reasoning-token budget and produced no correct answers on GSM8K. Evidence here does not support the hypothesis, but model quality is a confound.
- Confidence: Low, because the model is too weak; conclusions pertain only to small offline models.

## 7. Next Steps
1. Re-run with stronger API models (e.g., GPT-4.1, Claude Sonnet) to test hypothesis meaningfully and capture API token usage.
2. Add a lightweight regressor over question features + first-thought probe to predict token budget, then feed as a constraint.
3. Expand sample size (≥200) and include shorter-reasoning datasets (SVAMP) to test calibration across lengths.
4. Add enforcement: stop decoding when predicted budget is reached and measure accuracy loss vs. savings.
