# DeepSeek-R1-Distill-Llama-8B vs Llama 3 8B

A small benchmarking project comparing two 8B-parameter models on math, code and logic tasks:

| Model | Checkpoint used |
|-------|-----------------|
| **DeepSeek** | `unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit` |
| **Llama** | `unsloth/llama-3-8b-Instruct-bnb-4bit` |

Both models are loaded in 4-bit through [Unsloth](https://github.com/unslothai/unsloth). Each benchmark is sampled down to a few hundred prompts. Each prompt is run **3 times** per model, so every `responses/*.jsonl` file has 300 lines. The notebooks compute pass@1 and pass@3 style scores from those runs.

## Benchmarks

| Domain | Benchmark | Sample size | Source |
|--------|-----------|-------------|--------|
| Math | GSM8K (Socratic train split) | 100 | `grade-school-math/` |
| Math | Odyssey Math (competition-style problems with reasoning and levels) | 200 sampled, first 100 evaluated | `odyssey-math/` |
| Code | HumanEval | 100 | `human-eval/` |
| Code | MBPP (sanitized) | 100 | `retrieved_data/sanitized-mbpp.jsonl` |
| Logic | BIG-bench Navigate | 100 | `big_navigate/` |

## Repository layout

```
.
├── dataprep.ipynb        # 1. Sample each benchmark -> sampled_data/
├── benchmark.ipynb       # 2. Run a model over a sample -> responses/   (GPU, written for Kaggle)
├── evaluate.ipynb        # 3. Clean, grade and score responses -> evaluated_data/
├── template.jsonl        # Schema sketch for math and code records
├── retrieved_data/       # Raw benchmark files copied from upstream
├── sampled_data/         # Random samples produced by dataprep.ipynb
├── responses/            # Raw model generations (3 per prompt)
├── evaluated_data/       # Cleaned outputs, grades and execution results
```

The following upstream checkouts are **not included** in this repository (they are listed in `.gitignore` under `# Datasets`). You need to fetch them yourself if you want to rerun `dataprep.ipynb` or the code harness:

| Directory | Upstream |
|-----------|----------|
| `grade-school-math/` | https://github.com/openai/grade-school-math |
| `human-eval/` | https://github.com/openai/human-eval |
| `odyssey-math/` | https://github.com/protagolabs/odyssey-math |
| `big_navigate/` | Navigate task from https://github.com/google/BIG-bench (`task.json`) |

The data files this project actually depends on are copied into `retrieved_data/` and tracked, so the sampled data, responses and evaluation results can all be used without the checkouts.

## Pipeline

### 1. Data preparation (`dataprep.ipynb`)

Randomly samples each benchmark with `random.sample` and writes JSONL files to `sampled_data/`. Sampling is not seeded, so rerunning the notebook produces a different sample. Each benchmark has its own cell:

- **Odyssey**: flattens the nested JSONL, assigns `Odyssey/<n>` IDs, and keeps question, answer, reasoning and a `label - level` flag.
- **GSM8K**: parses the final answer after `####` into `ground_truth`.
- **MBPP**: extracts the function name and signature for `entry_point`. Builds a HumanEval-style `check(candidate)` test from `test_list`.
- **HumanEval**: copies `prompt`, `test`, `entry_point` and `canonical_solution`.
- **BIG Navigate**: keeps the input, instruction metadata and `target_scores` as `ground_truth`.

The MBPP and HumanEval cells read `sanitized-mbpp.jsonl` and `human-eval-v2-20210705.jsonl` from the working directory. Copies are in `retrieved_data/`.

### 2. Generation (`benchmark.ipynb`)

Written for a Kaggle/Colab notebook with a T4 GPU. It installs Unsloth, loads the model in 4-bit (`max_seq_length = 3072`), and reads a sample from `/kaggle/input/llmbenchmarks/`. Set `curr_benchmark` to pick the dataset. Each sample is repeated three times and truncated to 300 prompts. Generation runs in batches of 20 with `max_new_tokens = 2600`, using an Alpaca-style prompt (`### Instruction: ... ### Response:`). Output is written to `<model>_<benchmark>_responses.jsonl`.

To run Llama 3 instead of DeepSeek, swap the `model_name` (the Llama checkpoint is commented out in the notebook) and rename the output prefix. The Llama generation run is not saved as a separate notebook.

### 3. Evaluation (`evaluate.ipynb`)

Each benchmark is scored differently:

| Benchmark | Method |
|-----------|--------|
| GSM8K | Substring-match the ground-truth number in each output. Matches are then reviewed by hand, and the reviewer lists the row indices that were actually wrong (`manual` lists). The score is the share of the 300 attempts that are correct. |
| BIG Navigate | Strip DeepSeek's `<think>…</think>` block and extract the final answer. Look the answer up in `target_scores`. pass@1 picks one random attempt per prompt. pass@3 averages all attempts. |
| Odyssey | Pairs each response with the reference reasoning and answer. The pairs are written to `*_evaluated.jsonl`, graded 0–10 by an LLM judge (the judge output goes in `*_graded.jsonl`), and the `### Grade` line is parsed into `*_gradedfile.jsonl`. |
| HumanEval / MBPP | Extract the code block from each response (dropping DeepSeek's reasoning), then execute it against the tests with the HumanEval harness. Results go in `*_results.jsonl` with a `passed` flag. |

`evaluated_data/llama_mbpp.ipynb` is a scratch notebook used to inspect and repair individual MBPP completions.

## Results recorded in the notebooks

These are the numbers printed by `evaluate.ipynb` as last run. The sample is small and the runs are single-seed, so treat differences of a few points as noise.

| Benchmark | Metric | Llama 3 8B | DeepSeek-R1-Distill 8B |
|-----------|--------|-----------|------------------------|
| GSM8K | accuracy over 300 attempts | 0.653 | 0.687 |
| BIG Navigate | pass@1 | 0.55 | 0.74 |
| BIG Navigate | pass@3 (mean over attempts) | 0.587 | 0.793 |
| Odyssey | mean judge grade (0–10), pass@1 | 3.46 | 5.72 |
| Odyssey | mean judge grade (0–10), pass@3 | 3.54 | 5.55 |
| HumanEval | pass rate over 300 attempts | 0.45 | not recorded in repo |

Caveats:

- The Odyssey and Navigate score cells share a copy-pasted header, and the Odyssey DeepSeek cell prints "BIG Navigate". Read the cell's input path, not its printed title.
- The Llama MBPP results file reports 0% passed. This looks like a pipeline problem (formatting or encoding of completions), not a real model score. Treat MBPP as unfinished.
- DeepSeek HumanEval and MBPP execution results are not stored in the repo.

## Data formats

Sampled math record (`sampled_data/odyssey_sample.jsonl`, `gsm8k_sample.jsonl`):

```json
{"task_id": "GSM8K/12", "instruction": "...", "placeholder": "", "ground_truth": "42", "reasoning": "...", "flag": ""}
```

Sampled code record (`humaneval_sample.jsonl`, `mbpp_sample.jsonl`):

```json
{"task_id": "HumanEval/0", "prompt": "...", "placeholder": "", "test": "def check(candidate): ...", "entry_point": "f", "canonical_solution": "..."}
```

Response record (`responses/*.jsonl`): `{"task_id": ..., "output": ...}` for math/logic and `{"task_id": ..., "completion": ...}` for code. `template.jsonl` sketches the intended combined schema.

## Reproducing

1. Only needed to regenerate samples or run the code harness: clone the upstream repos from the table above into the repo root. The paths in `dataprep.ipynb` point at them (`odyssey-math/`, `grade-school-math/grade_school_math/data/`), or you can edit the `input_path` values to use the copies in `retrieved_data/`. Note that the Navigate cell reads `big-navigate/task.json` (hyphen), while the folder here is `big_navigate/` (underscore), so adjust one of them. The MBPP and HumanEval cells expect their input files in the working directory.
2. Run `dataprep.ipynb` to regenerate `sampled_data/`. To reproduce the reported numbers, use the committed samples instead, since sampling is unseeded.
3. Upload `sampled_data/` to a GPU environment (the notebook expects a Kaggle dataset named `llmbenchmarks`). Run `benchmark.ipynb` once per model and benchmark.
4. Put the generated `*_responses.jsonl` files in `responses/` and run the matching cells of `evaluate.ipynb`.
5. For HumanEval and MBPP, clone `human-eval` (see table above), install it (`pip install -e human-eval`) and run its `evaluate_functional_correctness` on the cleaned completions. Code execution is disabled by default in that harness. Read its README before enabling it, because it runs model-generated code.

Dependencies: Python 3.10+, `torch`, `unsloth`, `black` (code cleanup), plus the harness packages above. There is no `requirements.txt` for the top-level notebooks.

## Known limitations

- The 100-item samples and 3 generations per item give wide confidence intervals.
- GSM8K grading relies on substring matching plus hand review, and Odyssey grading relies on an LLM judge. Both are noisy.
- Context was capped at 3072 tokens (2600 for generation), which can truncate DeepSeek's long reasoning traces before the final answer.
- Paths in `evaluate.ipynb` are mixed Windows and POSIX style (`responses\llama_human-eval_responses.jsonl`). Some cells reference files such as `... copy.jsonl` that exist only locally, and a few cells define `rob_path` but use `prob_path`, so they depend on notebook state. Run cells in order.

## Licenses

Each upstream dataset keeps its own license (see the `LICENSE` file in `grade-school-math/`, `human-eval/` and `odyssey-math/`, and the BIG-bench terms in `big_navigate/README.md`). Model weights are governed by their respective Llama and DeepSeek licenses.
