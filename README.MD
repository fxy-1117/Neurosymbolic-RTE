# Neurosymbolic RTE Identification

Python code for the AAAI 2025 entailment / contradiction identification
experiments. The notebook logic is kept, with deterministic sampling plus logic
and neural caches.

## Environment

Use the `eesnli` conda environment. It should include the dependencies in
`requirements.txt`, plus:

- `transition_amr_parser`
- `amr_logic_converter`
- `en_core_web_sm`
- `AMR3-joint-ontowiki-seed43`

Check that the code starts:

```powershell
python -m rte_identification --help
```

## Data

Keep data under `data/`:

```text
data/esnli_train_1.csv
data/multinli_1.0/multinli_1.0_train.jsonl
```

SICK is loaded from the Hugging Face cache when available.

## Task List

Comment one task name and I will run it.

| Task | What it runs |
| --- | --- |
| `main` | e-SNLI, SICK, and MNLI main results |
| `esnli` | e-SNLI main result |
| `sick` | SICK main result |
| `mnli` | MNLI main result |
| `param` | e-SNLI parameter analysis |
| `noforget` | e-SNLI without forgetting |
| `noexpl` | e-SNLI without explanation |

Commands behind the tasks:

```powershell
python -m rte_identification paper --seed 66 --per-class 500 --threshold 0.55
python -m rte_identification run --dataset esnli --seed 66 --per-class 500
python -m rte_identification run --dataset sick --seed 66 --per-class 500
python -m rte_identification run --dataset mnli --seed 66 --per-class 500
python -m rte_identification parameter-analysis --dataset esnli --seed 66 --per-class 500 --max-len 20
python -m rte_identification run --dataset esnli --seed 66 --per-class 500 --no-forgetting
python -m rte_identification run --dataset esnli --seed 66 --per-class 500 --no-explanation
```

## Cache

- `.cache/logic/`: sentence-level AMR logic
- `.cache/neural/`: neural embeddings
- `.cache/nltk_data/`: local NLTK data

Outputs are written to `results/`.
