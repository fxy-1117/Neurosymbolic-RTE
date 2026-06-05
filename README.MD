# Neurosymbolic RTE Identification

This project contains the Python version of the AAAI 2025 entailment /
contradiction identification notebook.

The code keeps the original notebook reasoning logic, but moves experiment
execution into `.py` files with deterministic sampling, AMR logic caching, and
neural embedding caching.

## Code Layout

- `identification_repro/datasets.py`: dataset loading and fixed-seed sampling.
- `identification_repro/model_runtime.py`: AMR, spaCy, and neural scorer loaders.
- `identification_repro/reasoner.py`: notebook-derived symbolic prover.
- `identification_repro/experiments.py`: cache preparation, evaluation, and result JSON writing.
- `identification_repro/cli.py`: command line entry point.

## Environment

The experiments were run in the `eesnli` conda environment used by the original
notebook. Besides the dependencies in `requirements.txt`, the environment must
provide:

- `transition_amr_parser`
- `amr_logic_converter`
- the spaCy model `en_core_web_sm`
- the AMR checkpoint `AMR3-joint-ontowiki-seed43`

Run directly from the project directory:

```powershell
python -m identification_repro --help
```

Or install it locally:

```powershell
pip install -e .
```

## Data

Put local data files under `data/`, or leave them in the current directory:

```text
data/esnli_train_1.csv
data/multinli_1.0/multinli_1.0_train.jsonl
```

SICK is loaded from the Hugging Face datasets cache when available; otherwise
`datasets.load_dataset("sick")` is used.

## Reproduce Paper Tables

Run the three main datasets with the paper-style setting:

```powershell
python -m identification_repro paper --seed 66 --per-class 500 --threshold 0.55
```

Run one dataset:

```powershell
python -m identification_repro run --dataset esnli --seed 66 --per-class 500
python -m identification_repro run --dataset sick --seed 66 --per-class 500
python -m identification_repro run --dataset mnli --seed 66 --per-class 500
```

Outputs are written to `results/`.

## Parameter Analysis

The paper threshold sweep uses e-SNLI, 500 examples per class, max sentence
length 20, and thresholds `0.5 0.55 0.6 0.65 0.7 0.75 0.8 1.0`:

```powershell
python -m identification_repro parameter-analysis --dataset esnli --seed 66 --per-class 500 --max-len 20
```

The AMR logic cache is sentence-level and shared across datasets and thresholds,
so each unique sentence only needs to be parsed once.

## e-SNLI Ablations

Run the paper's e-SNLI ablations:

```powershell
$env:PYTHONPATH="src"
python -m identification_repro run --dataset esnli --seed 66 --per-class 500 --no-forgetting
python -m identification_repro run --dataset esnli --seed 66 --per-class 500 --no-explanation
```

## Cache

Two caches are used:

- `.cache/logic/sentence_logic.pkl`: AMR-to-logic formulas keyed by sentence text.
- `.cache/neural/`: SentenceTransformer embeddings used by predicate matching.
- `.cache/nltk_data/`: local NLTK resources such as WordNet.

To ignore the AMR logic cache and regenerate formulas for the current run:

```powershell
python -m identification_repro run --dataset esnli --no-logic-cache
```
