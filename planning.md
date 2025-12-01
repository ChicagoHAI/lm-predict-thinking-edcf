## Research Question
Can language models predict, before generation, how many tokens they will use to produce a response (their “thinking time”), and can this prediction guide decoding to improve efficiency without hurting accuracy?

## Background and Motivation
- Long reasoning traces (e.g., Chain-of-Thought, Tree-of-Thought, ReAct) vary widely in length, impacting latency and cost. Users lack visibility into how long a model will “think.”
- Prior work controls or samples reasoning length but seldom measures whether models can anticipate their own token budgets. Prediction could enable early-exit policies, budget-aware routing, and user-facing latency estimates.

## Hypothesis Decomposition
- H1: Given a problem statement, an LLM can estimate its required reasoning-token count with sublinear error (low MAE) relative to realized tokens under a fixed prompting strategy.
- H2: Conditioning decoding on a predicted budget (e.g., budget hint or early stopping) yields better accuracy-per-token than unguided decoding at similar or lower total tokens.
- Independent variables: prompting strategy (direct answer, greedy CoT), presence of budget-prediction prompt, decoding temperature.
- Dependent variables: answer accuracy (exact match), reasoning-token count, prediction error (MAE, calibration), accuracy-per-1k-tokens, overrun rate (predicted budget < realized).
- Success criteria: MAE < 25% of mean chain length; accuracy-per-token improves over baselines; no >3% absolute accuracy drop versus standard CoT.

## Proposed Methodology

### Approach
- Use GSM8K as the controlled reasoning benchmark (locally available).
- Build a small evaluation harness that queries real LLM APIs (OpenRouter key available) to:
  1) Ask the model to **predict** the number of reasoning tokens it will use (budget estimate).
  2) Generate the answer with or without the budget hint.
- Compare against baselines: direct answer (no rationale), greedy CoT without prediction, and CoT with self-reported budget.
- Measure prediction calibration and efficiency (accuracy per token).

### Experimental Steps
1. **Data prep**: Load GSM8K, create a manageable eval subset (e.g., 100 examples) for cost/time; log question lengths.
2. **Prompt design**: Two prompts—(a) CoT baseline; (b) Budget-prediction prompt that asks for a token-count estimate before solving (or provides a budget hint).
3. **API harness**: Call a strong model (e.g., `gpt-4.1` via OpenRouter) with deterministic settings (temperature 0–0.2) for stability; record token usage from API metadata.
4. **Prediction extraction**: Parse the model’s predicted token count from structured output; compute realized reasoning tokens (count rationale tokens prior to final answer).
5. **Evaluation**: Metrics per example—accuracy (string match on final answer), predicted vs. realized tokens (MAE, bias), accuracy-per-1k-tokens, overrun rate.
6. **Analysis**: Correlate predicted vs. realized tokens; compare efficiency against baselines; conduct paired tests on token counts and accuracy.

### Baselines
- **Direct answer**: Single-turn answer, no rationale (minimal tokens).
- **Greedy CoT**: Standard few-shot CoT prompt, no prediction.
- **Prediction-aware CoT**: Model states predicted token count, then reasons; optionally condition with a “stay within N tokens” hint.

### Evaluation Metrics
- Exact match accuracy on GSM8K subset.
- Token metrics: total reasoning tokens, MAE and bias of predicted vs. realized tokens, overrun rate (% cases realized > predicted).
- Efficiency: accuracy-per-1k reasoning tokens.
- Calibration: expected calibration error (ECE) over normalized prediction error bins.

### Statistical Analysis Plan
- Paired comparisons across conditions (same questions): Wilcoxon signed-rank for non-normal token errors; paired t-test for accuracy-per-token if approximately normal.
- Report 95% bootstrap CIs for accuracy and MAE (1k bootstrap samples).
- Correlation (Pearson/Spearman) between predicted and realized tokens.

## Expected Outcomes
- Support H1 if predictions correlate well with realized tokens (ρ > 0.5) and MAE is modest.
- Support H2 if prediction-aware CoT achieves equal/greater accuracy with fewer tokens or higher accuracy-per-token than greedy CoT.
- Refute if predictions are uncorrelated or efficiency degrades.

## Timeline and Milestones
- Planning (done): 20 min.
- Harness + prompts + small eval set: 45–60 min.
- Run experiments (100 examples): 45–60 min depending on API latency.
- Analysis + plots: 45 min.
- Documentation (REPORT.md, README.md): 30 min.

## Potential Challenges
- API variability or rate limits → add retry/backoff and small batch sizes.
- Parsing predicted token counts robustly → enforce JSON output and validation.
- Token accounting: need to separate reasoning tokens from final answer → require “Reasoning:” section and count tokens via tokenizer.
- Cost/time: cap eval to ~100 examples initially; can scale if time permits.

## Success Criteria
- Completed runs on the GSM8K subset with recorded predicted and realized tokens.
- Statistical comparison showing calibrated predictions and efficiency gains or a clear negative result documented.
- Reproducible harness with seeds, prompts, and saved outputs in `results/`.
