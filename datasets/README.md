# Downloaded Datasets

This directory hosts datasets used to study whether language models can predict their own thinking tokens. Large payloads are excluded from git via `.gitignore`; follow the download notes below to reproduce.

## Dataset 1: GSM8K (Grade School Math)

### Overview
- **Source**: https://github.com/openai/grade-school-math (mirrored on HuggingFace as `openai/gsm8k`)
- **Size**: 7,473 train problems, 1,319 test problems (~13 MB total)
- **Format**: JSON Lines; fields `question` and `answer` (worked solutions)
- **Task**: Math word-problem solving; prompts typically require chain-of-thought reasoning
- **Splits**: train (7,473), test (1,319)
- **License**: MIT (per upstream repo)

### Download Instructions

**Direct (used here):**
```bash
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl -O datasets/gsm8k_train.jsonl
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -O datasets/gsm8k_test.jsonl
```

**HuggingFace alternative:**
```python
from datasets import load_dataset
hf_ds = load_dataset("openai/gsm8k")
hf_ds.save_to_disk("datasets/gsm8k_hf")
```

### Loading the Dataset
```python
import json
with open("datasets/gsm8k_train.jsonl") as f:
    records = [json.loads(line) for line in f]
print(records[0]["question"], "->", records[0]["answer"])
```

### Sample Data
See `datasets/samples/gsm8k_sample.json` for 5 example records (question + worked solution).

### Notes
- Well-suited for testing whether models can predict the number of reasoning tokens needed (short vs. long solutions).
- Combine with adaptive decoding (e.g., selective CoT) to measure prediction accuracy vs. actual generation cost.
