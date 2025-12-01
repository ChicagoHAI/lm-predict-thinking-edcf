# Cloned Repositories

## Repo 1: tree-of-thought-llm
- **URL**: https://github.com/princeton-nlp/tree-of-thought-llm
- **Purpose**: Official Tree-of-Thought implementation for adaptive reasoning search; useful baseline for varying reasoning token counts.
- **Location**: code/tree-of-thought-llm/
- **Key files**: `cot-tree-search.ipynb`, `src/` utilities for BFS/DFS over reasoning traces.
- **Notes**: Requires Python, OpenAI API for some notebooks; check README for environment setup.

## Repo 2: ReAct
- **URL**: https://github.com/ysymyth/ReAct
- **Purpose**: Combines reasoning traces with actions; provides baselines for reasoning efficiency and introspection.
- **Location**: code/react/
- **Key files**: `experiments/` prompts, `react/generic.py` agent loop, README instructions.
- **Notes**: Uses OpenAI/PaLM APIs; adapt prompts to instrument token counts.
