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

Data files are not included in the GitHub code because they are large. To run
from scratch, create `data/` locally and put the required files there:

```text
data/esnli_train_1.csv
data/multinli_1.0/multinli_1.0_train.jsonl
```

Only `esnli_train_1.csv` is needed for the e-SNLI experiments in this code.
Other e-SNLI files such as dev/test/train_2 are optional and are not used by
the default tasks.

SICK is loaded from the Hugging Face cache when available.

## Run

In VSCode, open `run.py` and click Run. Or start the task list from a terminal:

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

Cache and result files are local outputs. They are not included in the GitHub
code.

- `.cache/logic/`: sentence-level AMR logic
- `.cache/neural/`: neural embeddings
- `.cache/nltk_data/`: local NLTK data

Outputs are written to `results/`.
