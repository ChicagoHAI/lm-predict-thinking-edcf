## Resources Catalog

### Summary
Collected papers, one benchmark dataset, and two reference codebases to study whether LMs can predict their own reasoning token budgets.

### Papers
Total papers downloaded: 6

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Chain-of-Thought Prompting Elicits Reasoning in LLMs | Wei et al. | 2022 | papers/2201.11903_chain-of-thought.pdf | Introduces CoT prompting baseline. |
| Self-Consistency Improves Chain of Thought Reasoning | Wang et al. | 2023 | papers/2203.11171_self-consistency.pdf | Samples multiple rationales; more tokens vs accuracy. |
| Least-to-Most Prompting Enables Complex Reasoning | Zhou et al. | 2023 | papers/2205.10625_least-to-most.pdf | Ordered decomposition controls chain length. |
| ReAct: Synergizing Reasoning and Acting in LLMs | Yao et al. | 2023 | papers/2210.03629_react.pdf | Interleaves thoughts and tool actions. |
| Reflexion: Language Agents with Verbal RL | Shinn et al. | 2023 | papers/2303.11366_reflexion.pdf | Self-evaluation to adapt reasoning depth. |
| Tree of Thoughts: Deliberate Problem Solving with LLMs | Yao et al. | 2023 | papers/2305.10601_tree-of-thought.pdf | Search over reasoning branches with budgets. |

### Datasets
Total datasets downloaded: 1

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K | OpenAI grade-school-math | 7.4k train / 1.3k test (~13 MB) | Math word-problem reasoning | datasets/gsm8k_train.jsonl, datasets/gsm8k_test.jsonl | Includes samples in `datasets/samples/gsm8k_sample.json`; good for token-length variance. |

### Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| tree-of-thought-llm | https://github.com/princeton-nlp/tree-of-thought-llm | Official Tree-of-Thought implementation for controllable reasoning search | code/tree-of-thought-llm/ | Notebooks and Python utilities; requires API keys. |
| ReAct | https://github.com/ysymyth/ReAct | Reasoning + acting agents baseline | code/react/ | Prompt templates and agent loop; adapt to instrument token counts. |

### Resource Gathering Notes
- **Search Strategy**: Pulled seminal reasoning-efficiency papers (CoT, self-consistency, structured decomposition, reasoning-action, search-based) and matched them with accessible implementations; chose GSM8K for math reasoning token variability.
- **Selection Criteria**: Emphasized methods that vary or control reasoning length, have open PDFs, and have usable codebases.
- **Challenges Encountered**: arXiv API search rate-limited; relied on known arXiv IDs for direct PDF fetches.
- **Gaps and Workarounds**: Only one dataset downloaded to keep footprint small; add SVAMP/MultiArith later if needed.

### Recommendations for Experiment Design
1. **Primary dataset(s)**: GSM8K; optionally add SVAMP for shorter chains to test prediction calibration across lengths.
2. **Baseline methods**: Direct answer, greedy CoT, self-consistency (k=5), Tree-of-Thought (depth 2–4), ReAct agent with/without reflection.
3. **Evaluation metrics**: Accuracy, reasoning-token count, MAE of predicted vs. actual tokens, accuracy-per-token, and overrun rate (% examples exceeding budget).
4. **Code to adapt/reuse**: Use Tree-of-Thought repo for controllable depth/breadth search; ReAct repo for action-reasoning traces; simple GSM8K loader from dataset README.
