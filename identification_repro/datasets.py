"""Dataset loading and deterministic sampling for the paper experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


LABELS = ["ent", "con", "neu"]
NOTEBOOK_SAMPLE_ORDER = ["ent", "neu", "con"]


@dataclass(frozen=True)
class Example:
    """One NLI example in the notebook format.

    sentences contains premise/claim for SICK and MultiNLI, and
    premise/claim/explanation for e-SNLI.
    """

    sentences: tuple[str, ...]
    label: str

    def as_notebook_row(self) -> list[str]:
        return list(self.sentences) + [self.label]


def _find_data_file(data_dir: Path, filename: str) -> Path:
    candidates = [data_dir / filename, Path(filename)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in sorted(data_dir.rglob(filename)):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in {data_dir} or the current directory")


def _within_max_len(sentences: Iterable[str], max_len: int) -> bool:
    if max_len <= 0:
        return True
    return all(len(word_tokenize(sentence)) <= max_len for sentence in sentences)


def sample_balanced(
    buckets: Dict[str, List[Example]],
    per_class: int,
    seed: int,
    label_order: list[str] = NOTEBOOK_SAMPLE_ORDER,
) -> List[Example]:
    """Sample examples with the same RNG style and label order as the notebook."""
    np.random.seed(seed)

    selected: List[Example] = []
    for label in label_order:
        if len(buckets[label]) < per_class:
            raise ValueError(f"Only found {len(buckets[label])} {label} examples, need {per_class}")
        indices = np.random.choice(len(buckets[label]), per_class, replace=False)
        selected.extend(buckets[label][int(index)] for index in indices)
        print(f"[data] {label}: selected {per_class} of {len(buckets[label])}", flush=True)
    return selected


def load_esnli(
    data_dir: Path,
    per_class: int,
    seed: int,
    max_len: int,
    include_explanation: bool = True,
) -> List[Example]:
    """Load the e-SNLI train split used by the identification notebook."""
    path = _find_data_file(data_dir, "esnli_train_1.csv")
    df = pd.read_csv(path)
    buckets: Dict[str, List[Example]] = {"ent": [], "con": [], "neu": []}
    label_map = {"entailment": "ent", "contradiction": "con", "neutral": "neu"}

    print("[data] scanning e-SNLI train split", flush=True)
    rows = df.iloc[: min(259999, len(df)), [1, 2, 3, 4]]
    for label_raw, premise, claim, explanation in rows.itertuples(index=False, name=None):
        if not all(isinstance(x, str) for x in (premise, claim, explanation)):
            continue
        sentences = (premise, claim, explanation) if include_explanation else (premise, claim)
        if not _within_max_len(sentences, max_len):
            continue
        label = label_map.get(label_raw)
        if label:
            buckets[label].append(Example(sentences, label))

    return sample_balanced(buckets, per_class, seed)


def _load_sick_splits():
    """Load SICK in the same order as the notebook reproduction runner.

    The local HuggingFace Arrow cache is discovered with a sorted glob in the
    notebook-derived runner. On this machine that order is test, train,
    validation; keeping it here makes fixed-seed samples identical.
    """
    from datasets import Dataset, load_dataset

    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "sick"
    arrow_files = sorted(cache_root.glob("default/0.0.0/*/sick-*.arrow"))
    if arrow_files:
        print("[data] loading SICK from cached Arrow files", flush=True)
        return [Dataset.from_file(str(path)) for path in arrow_files]

    return [
        load_dataset("sick", split="train"),
        load_dataset("sick", split="test"),
        load_dataset("sick", split="validation"),
    ]


def load_sick(data_dir: Path, per_class: int, seed: int, max_len: int) -> List[Example]:
    """Load SICK examples with normalized entailment_AB relation strings."""
    del data_dir
    buckets: Dict[str, List[Example]] = {"ent": [], "con": [], "neu": []}

    print("[data] scanning SICK splits", flush=True)
    for split in _load_sick_splits():
        for row in split:
            sentence_a = row["sentence_A"]
            sentence_b = row["sentence_B"]
            if not all(isinstance(x, str) for x in (sentence_a, sentence_b)):
                continue
            if not _within_max_len((sentence_a, sentence_b), max_len):
                continue

            relation = str(row["entailment_AB"]).strip()

            if row["label"] == 0:
                if relation == "A_entails_B":
                    buckets["ent"].append(Example((sentence_a, sentence_b), "ent"))
                else:
                    buckets["ent"].append(Example((sentence_b, sentence_a), "ent"))
            elif row["label"] == 1:
                if relation == "A_neutral_B":
                    buckets["neu"].append(Example((sentence_a, sentence_b), "neu"))
                else:
                    buckets["neu"].append(Example((sentence_b, sentence_a), "neu"))
            elif row["label"] == 2:
                if relation == "A_contradicts_B":
                    buckets["con"].append(Example((sentence_a, sentence_b), "con"))
                else:
                    buckets["con"].append(Example((sentence_b, sentence_a), "con"))

    return sample_balanced(buckets, per_class, seed)


def load_mnli(data_dir: Path, per_class: int, seed: int, max_len: int) -> List[Example]:
    """Load MultiNLI train examples from multinli_1.0_train.jsonl."""
    path = _find_data_file(data_dir, "multinli_1.0_train.jsonl")
    buckets: Dict[str, List[Example]] = {"ent": [], "con": [], "neu": []}
    label_map = {"entailment": "ent", "contradiction": "con", "neutral": "neu"}

    print("[data] scanning MultiNLI train split", flush=True)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sentence1 = row["sentence1"]
            sentence2 = row["sentence2"]
            if not all(isinstance(x, str) for x in (sentence1, sentence2)):
                continue
            if not _within_max_len((sentence1, sentence2), max_len):
                continue
            label = label_map.get(row["gold_label"], "neu")
            buckets[label].append(Example((sentence1, sentence2), label))

    return sample_balanced(buckets, per_class, seed)


def load_dataset_examples(
    name: str,
    data_dir: Path,
    per_class: int,
    seed: int,
    max_len: int,
    include_esnli_explanation: bool = True,
) -> List[Example]:
    if name == "esnli":
        return load_esnli(data_dir, per_class, seed, max_len, include_esnli_explanation)
    if name == "sick":
        return load_sick(data_dir, per_class, seed, max_len)
    if name == "mnli":
        return load_mnli(data_dir, per_class, seed, max_len)
    raise ValueError(f"Unknown dataset: {name}")
