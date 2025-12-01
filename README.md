## Project Overview
Research workspace exploring whether LMs can predict their own reasoning-token budgets (“thinking time”) on GSM8K math problems using a local Qwen2.5-0.5B-Instruct model.

## Key Findings
- Prediction-aware prompting produced weak calibration: MAE≈198 tokens, correlation ≈0.14 between predicted and actual reasoning tokens.
- 83% of cases exceeded the stated budget; some predictions were extreme (e.g., 1880 tokens).
- Task accuracy was 0 on a 20-example GSM8K subset, so efficiency gains were not observable with this small model.
- Visuals and metrics: see `results/plots/` and `results/metrics.json`.

## Reproduction
```bash
uv venv
source .venv/bin/activate
uv sync
python -m research_workspace.experiment
```
Outputs: `results/metrics.json`, `results/raw/token_prediction_runs.parquet`, plots in `results/plots/`.

## File Structure
- `planning.md` – research plan.
- `src/research_workspace/experiment.py` – experiment harness (prompts, parsing, metrics, plots).
- `datasets/` – GSM8K JSONL files (local).
- `results/` – metrics, raw run data, plots.
- `REPORT.md` – full report with analysis and next steps.

## More Detail
See `REPORT.md` for methodology, full metrics tables, limitations, and suggested follow-ups with stronger models.
