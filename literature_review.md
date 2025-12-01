## Literature Review

### Research Area Overview
Predicting a language model’s own “thinking tokens” (i.e., the number of generated reasoning steps) relates to controllable chain-of-thought (CoT) generation, adaptive decoding, and introspective estimation of compute. Recent work explores structured reasoning traces (CoT, Least-to-Most), sampling-based exploration (Self-Consistency), reasoning-action loops (ReAct), and search-based planning (Tree-of-Thought). These methods vary reasoning length and therefore offer natural testbeds for studying whether a model can forecast its required token budget.

### Key Papers

#### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022, arXiv:2201.11903)
- **Key Contribution**: Shows that few-shot CoT prompts unlock complex reasoning in large models.
- **Methodology**: Provide worked reasoning demonstrations; greedy decoding produces multi-step rationales.
- **Datasets Used**: GSM8K, MultiArith, SVAMP, etc.
- **Results**: Large gains over standard prompting (e.g., >50% absolute on GSM8K for PaLM 540B).
- **Code Available**: Not official; many community reimplementations.
- **Relevance**: Establishes baseline reasoning length distributions to predict against.

#### Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2023, arXiv:2203.11171)
- **Key Contribution**: Samples multiple CoT paths and majority-votes answers, boosting accuracy.
- **Methodology**: Diverse sampling (temperature >0), then marginalize over sampled rationales.
- **Datasets Used**: GSM8K, AQuA, StrategyQA, SVAMP.
- **Results**: +6–10% absolute accuracy over greedy CoT with modest extra tokens.
- **Code Available**: Reference code in paper supplement.
- **Relevance**: Provides paired (predicted vs. realized) reasoning-length statistics under different sample counts.

#### Least-to-Most Prompting Enables Complex Reasoning in Large Language Models (Zhou et al., 2023, arXiv:2205.10625)
- **Key Contribution**: Decomposes hard problems into ordered subproblems to improve generalization.
- **Methodology**: Prompt the model to generate sub-questions and solve sequentially; reasoning length becomes explicit and controllable.
- **Datasets Used**: GSM8K, Colored Objects, Coin Flips.
- **Results**: Outperforms standard CoT on compositional tasks.
- **Code Available**: Demonstrations in paper appendix; community repos exist.
- **Relevance**: Offers a knob to vary reasoning depth for prediction experiments.

#### ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023, arXiv:2210.03629)
- **Key Contribution**: Interleaves reasoning traces with environment actions (e.g., search, tools).
- **Methodology**: Prompt format alternates “Thought” and “Action/Observation” tokens; evaluates on QA and interactive tasks.
- **Datasets Used**: HotpotQA, ALFWorld, WebShop.
- **Results**: Improves success rates over pure reasoning or pure acting baselines.
- **Code Available**: Official GitHub (cloned here).
- **Relevance**: Produces variable-length reasoning-action traces; good for measuring predicted vs. actual token budgets in interactive settings.

#### Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023, arXiv:2303.11366)
- **Key Contribution**: Adds self-evaluation and reflection to refine subsequent reasoning attempts.
- **Methodology**: After each trial, the model summarizes mistakes and proposes fixes; repeats until success or budget limit.
- **Datasets Used**: HotpotQA, FEVER, AlfWorld-like tasks.
- **Results**: Significant performance boosts with limited additional attempts.
- **Code Available**: Reference implementations; agent loop straightforward to re-create.
- **Relevance**: Provides adaptive token use and self-prediction signals (reflection notes) for estimating required thinking time.

#### Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023, arXiv:2305.10601)
- **Key Contribution**: Searches over reasoning branches instead of a single chain.
- **Methodology**: BFS/DFS over partial thoughts with value functions guiding pruning; configurable depth/width budgets.
- **Datasets Used**: Game of 24, creative writing, mini crosswords.
- **Results**: Outperforms linear CoT by exploring multiple branches while controlling token budgets.
- **Code Available**: Official GitHub (cloned here).
- **Relevance**: Enables explicit budget-aware inference; ideal for benchmarking prediction of tokens versus chosen search depth.

### Common Methodologies
- **Few-shot CoT prompting** (Wei et al.): base approach for multi-step rationales.
- **Sampling-based diversity and voting** (Self-Consistency): multiple sampled chains improve robustness.
- **Structured decomposition** (Least-to-Most): enforce ordered sub-steps to control reasoning depth.
- **Reasoning-action interleaving** (ReAct, Reflexion): add tool calls and self-feedback, inducing variable token counts.
- **Search over thoughts** (Tree-of-Thought): explicit branching with tunable breadth/depth budgets.

### Standard Baselines
- **Direct Answer Prompting**: single-step answer generation without rationale; minimal tokens.
- **Greedy CoT**: single reasoning chain via greedy decoding; medium token cost.
- **Self-Consistency (k samples)**: k sampled rationales then vote; token cost roughly k× chain length.
- **Program-of-Thought / Toolformer baselines**: emit code or tool calls instead of long rationales (shorter tokens, external compute).

### Evaluation Metrics
- **Answer accuracy / exact match** on task datasets (e.g., GSM8K EM).
- **Token budget metrics**: total reasoning tokens, max tokens per example, ratio of predicted vs. realized tokens.
- **Efficiency metrics**: accuracy-per-token, time-to-first-correct.
- **Calibration metrics**: Brier score / ECE for predicted token counts or success probabilities.

### Datasets in the Literature
- **GSM8K**: primary math reasoning benchmark (used by CoT, Self-Consistency, Least-to-Most).
- **SVAMP / MultiArith**: simpler math word problems for shorter reasoning chains.
- **HotpotQA / FEVER**: open-domain QA with multi-hop reasoning (ReAct, Reflexion).
- **Game of 24 / puzzles**: token-depth-sensitive tasks used by Tree-of-Thought.

### Gaps and Opportunities
- Few papers directly predict reasoning length; most set fixed budgets. Opportunities include learning a token-count regressor from CoT traces and using it to gate sampling.
- Limited exploration of real-time token forecasting under tool-augmented settings (ReAct/Reflexion).
- Lack of standardized efficiency metrics (accuracy per token) across reasoning strategies.

### Recommendations for Our Experiment
- **Recommended datasets**: GSM8K (available locally), plus light-weight samples of SVAMP/MultiArith if needed for short-chain regimes.
- **Recommended baselines**: direct answer vs. greedy CoT vs. self-consistency (k=5) vs. Tree-of-Thought (depth-limited); compare predicted vs. actual token counts.
- **Recommended metrics**: accuracy, token count per example, over/under-prediction error (MAE), and accuracy-per-1k-tokens.
- **Methodological considerations**: log token counts from API/decoder; train a small predictor using prompt features (question length, complexity heuristics) and optionally one-step probe outputs to forecast required thinking tokens; evaluate early-exit policies guided by predictions.
