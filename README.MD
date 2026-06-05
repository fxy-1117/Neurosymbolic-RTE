# Neurosymbolic RTE Identification

Python code for entailment / contradiction identification experiments. The
notebook logic is kept, with deterministic sampling plus logic and neural
caches.

## Environment

Use the `eesnli` conda environment. It should include the dependencies in
`requirements.txt`, plus:

- `transition_amr_parser`
- `amr_logic_converter`
- `en_core_web_sm`
- `AMR3-joint-ontowiki-seed43`

## Data

Keep data under `data/`:

```text
data/esnli_train_1.csv
data/multinli_1.0/multinli_1.0_train.jsonl
```

SICK is loaded from the Hugging Face cache when available.

## Run

Start the task list:

```powershell
python -m rte_identification
```

Then choose one or more tasks by number or name.

| Task | What it runs |
| --- | --- |
| `main` | e-SNLI, SICK, and MNLI main results |
| `esnli` | e-SNLI main result |
| `sick` | SICK main result |
| `mnli` | MNLI main result |
| `param` | e-SNLI parameter analysis |
| `noforget` | e-SNLI without forgetting |
| `noexpl` | e-SNLI without explanation |

You can also skip the prompt:

```powershell
python -m rte_identification tasks --select main
python -m rte_identification tasks --select esnli,param
```

## Cache

- `.cache/logic/`: sentence-level AMR logic
- `.cache/neural/`: neural embeddings
- `.cache/nltk_data/`: local NLTK data

Outputs are written to `results/`.
