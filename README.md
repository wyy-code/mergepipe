# MergePipe

**MergePipe** is a budget-aware checkpoint merging system for large language models.
It enables efficient and controllable merging of base and expert models under
explicit expert I/O budgets, with full plan / execution lineage tracking.

MergePipe is designed for:
- Model merging at scale
- Budget-constrained expert integration
- Reproducible, auditable merge pipelines

---

## Key Features

- **Budget-aware merging**
  - Explicit expert I/O budget (MB)
  - Block-level planning and execution
- **Pluggable merge strategies**
  - `ties`, `avg`, `dare`
- **Sparse output support**
  - Top-p masking to reduce write cost
- **Deterministic planning**
  - Plan digest with hash / ID
- **Full lineage & manifest**
  - Run-level metadata for reproducibility

---

## Installation

```bash
git clone <this-repo>
cd mergepipe
pip install -e .
````

Python >= 3.9 is recommended.

---

## Quick Start

### 1. Prepare MergeDB

MergePipe uses a SQLite database (`mergedb.sqlite`) to track base and expert models.

Ensure your database contains:

* a base model entry
* one or more expert model entries

---

### 2. Run a Merge

```bash
mergepipe \
  --db /path/to/mergedb.sqlite \
  --base qwen3_base \
  --experts qwen3_distill \
  --out /path/to/output_dir \
  --model-id qwen3-0.6B
```

On success, you should see output like:

```text
[OK] merged base=qwen3_base experts=['qwen3_distill']
 -> /path/to/output_dir
 | run=runs/20260203_XXXXXX_qwen3-0.6B
 | plan=0.04s exec=15.8s
```

---

## Common Options

| Option           | Description                            | Default  |
| ---------------- | -------------------------------------- | -------- |
| `--strategy`     | Merge strategy (`ties`, `avg`, `dare`) | `ties`   |
| `--scoring`      | Planner scoring function               | `l2cos`  |
| `--io-budget-mb` | Expert I/O budget (MB)                 | `906`    |
| `--top-p`        | Top-p for sparse output                | `0.35`   |
| `--block-size`   | Block size (elements)                  | `640000` |
| `--device`       | Execution device                       | `cpu`    |

---

## Output Structure

```
output_dir/
├── config.json
├── model.safetensors / pytorch_model.bin
├── tokenizer.json
├── tokenizer.model
└── ...
```

Run artifacts are stored under:

```
runs/
└── <run_id>/
    ├── plan.json
    ├── manifest.json
    └── meta.json
```

---

## Reproducibility

Each merge run records:

* planner inputs and outputs
* execution statistics
* I/O usage
* plan hash and run ID

This allows exact replay and audit of merge results.
