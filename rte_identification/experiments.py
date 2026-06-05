"""Experiment runner for the identification notebook experiments."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from . import reasoner
from .datasets import Example, LABELS, load_dataset_examples
from .model_runtime import NeuralScorer, ensure_nltk_data, load_amr_runtime, load_spacy_model


LOGIC_CACHE_VERSION = 1


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    per_class: int = 500
    seed: int = 66
    max_len: int = 0
    threshold: float = 0.55
    batch_size: int = 256
    roberta_batch_size: int = 256
    use_forgetting: bool = True
    include_explanation: bool = True

    def mode_suffix(self) -> str:
        parts = []
        if not self.use_forgetting:
            parts.append("noforget")
        if not self.include_explanation:
            parts.append("noexpl")
        return "" if not parts else "_" + "_".join(parts)

    def slug(self) -> str:
        max_len = self.max_len if self.max_len > 0 else "none"
        threshold = str(self.threshold).replace(".", "p")
        return (
            f"{self.dataset}_seed{self.seed}_n{self.per_class}"
            f"_maxlen{max_len}_thr{threshold}_bs{self.batch_size}"
            f"{self.mode_suffix()}"
        )


def examples_fingerprint(examples: Sequence[Example]) -> str:
    payload = json.dumps([asdict(example) for example in examples], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _logic_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "logic" / "sentence_logic.pkl"


def _neural_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "neural" / "sentence_transformer_tensor_embeddings.pkl"


def load_sentence_logic_cache(path: Path) -> dict[str, object]:
    """Load the sentence-level AMR logic cache.

    The cache is keyed only by sentence text so different datasets, thresholds,
    and ablation modes can reuse the same AMR-to-logic conversion.
    """
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        print(f"[cache] ignoring incompatible logic cache: {path}", flush=True)
        return {}
    if payload.get("version") != LOGIC_CACHE_VERSION:
        print(f"[cache] ignoring old logic cache version: {path}", flush=True)
        return {}
    logic_by_sentence = payload.get("logic_by_sentence", {})
    print(f"[cache] loaded {len(logic_by_sentence)} sentence logic entries: {path}", flush=True)
    return logic_by_sentence


def save_sentence_logic_cache(path: Path, logic_by_sentence: dict[str, object]) -> None:
    """Persist sentence logic after each generated batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "version": LOGIC_CACHE_VERSION,
                "logic_by_sentence": logic_by_sentence,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[cache] saved {len(logic_by_sentence)} sentence logic entries: {path}", flush=True)


def unique_sentences(examples: Sequence[Example]) -> list[str]:
    """Return unique sentences in first-seen order."""
    return list(dict.fromkeys(sentence for example in examples for sentence in example.sentences))


def generate_sentence_logic_chunks(
    sentences: Sequence[str],
    batch_size: int,
    roberta_batch_size: int,
) -> Iterable[dict[str, object]]:
    """Generate AMR logic for uncached sentences in large parser batches."""
    if not sentences:
        return

    parser, converter = load_amr_runtime()
    total = len(sentences)

    for start in range(0, total, batch_size):
        chunk = list(sentences[start : start + batch_size])
        token_batch = []
        for sentence in chunk:
            tokens, _positions = parser.tokenize(sentence)
            token_batch.append(tokens)

        end = start + len(chunk)
        print(
            f"[logic] parsing uncached sentences {start + 1}-{end} of {total}, "
            f"batch_size={batch_size}, roberta_batch_size={roberta_batch_size}",
            flush=True,
        )
        annotations, _machines = parser.parse_sentences(
            token_batch,
            batch_size=batch_size,
            roberta_batch_size=roberta_batch_size,
        )

        print(f"[logic] converting sentences {start + 1}-{end} to logic formulas", flush=True)
        converted = [converter.convert(annotation) for annotation in annotations]
        yield dict(zip(chunk, converted))


def logic_for_examples(
    examples: Sequence[Example],
    cache_path: Path,
    batch_size: int,
    roberta_batch_size: int,
    reuse_logic_cache: bool,
):
    """Return notebook-shaped logic rows, generating missing sentence cache entries."""
    logic_by_sentence = load_sentence_logic_cache(cache_path) if reuse_logic_cache else {}
    sentences = unique_sentences(examples)
    missing_sentences = [sentence for sentence in sentences if sentence not in logic_by_sentence]

    print(
        f"[cache] sentence logic hits={len(sentences) - len(missing_sentences)} "
        f"misses={len(missing_sentences)} total_unique={len(sentences)}",
        flush=True,
    )
    if missing_sentences:
        for generated in generate_sentence_logic_chunks(
            missing_sentences,
            batch_size=batch_size,
            roberta_batch_size=roberta_batch_size,
        ):
            logic_by_sentence.update(generated)
            save_sentence_logic_cache(cache_path, logic_by_sentence)

    return [[logic_by_sentence[sentence] for sentence in example.sentences] for example in examples]


def run_prover(examples: Sequence[Example], logic_data) -> dict:
    """Run the symbolic prover and collect the same metrics printed by the notebook."""
    predicted: List[str] = []
    gold: List[str] = []
    skipped_both = 0
    errors = []

    for index, (logic, example) in enumerate(tqdm(list(zip(logic_data, examples)), desc="[prove]")):
        try:
            label = reasoner.prove(logic[:], example.as_notebook_row())
            if label == "both":
                skipped_both += 1
                continue
            predicted.append(label)
            gold.append(example.label)
        except Exception as exc:
            errors.append({"index": index, "type": type(exc).__name__, "message": str(exc)})
            print(f"[prove] exception at {index}: {type(exc).__name__}: {exc}", flush=True)

    report_text = classification_report(gold, predicted, zero_division=0)
    report = classification_report(gold, predicted, output_dict=True, zero_division=0)
    matrix = confusion_matrix(gold, predicted, labels=LABELS).tolist()
    accuracy = accuracy_score(gold, predicted)

    print(f"[prove] kept={len(gold)} skipped_both={skipped_both} errors={len(errors)}", flush=True)
    print(f"Accuracy: {accuracy:.6f}", flush=True)
    print(report_text, flush=True)
    print(f"Confusion matrix labels={LABELS}:\n{matrix}", flush=True)

    return {
        "gold": gold,
        "predicted": predicted,
        "skipped_both": skipped_both,
        "errors": errors,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def prepare_core(cache_dir: Path, threshold: float, use_forgetting: bool = True) -> NeuralScorer:
    ensure_nltk_data()
    print("[runtime] loading SentenceTransformer scorer", flush=True)
    scorer = NeuralScorer(cache_path=_neural_cache_path(cache_dir))
    print("[runtime] loading spaCy model", flush=True)
    spacy_model = load_spacy_model()
    reasoner.configure_runtime(
        scorer=scorer,
        spacy_model=spacy_model,
        threshold=threshold,
        use_forgetting=use_forgetting,
    )
    print("[runtime] neural scorer and spaCy model ready", flush=True)
    return scorer


def run_experiment(
    config: ExperimentConfig,
    data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    reuse_logic_cache: bool = True,
) -> dict:
    """Run one dataset setting and write a result JSON file."""
    ensure_nltk_data()
    examples = load_dataset_examples(
        config.dataset,
        data_dir=data_dir,
        per_class=config.per_class,
        seed=config.seed,
        max_len=config.max_len,
        include_esnli_explanation=config.include_explanation,
    )
    fingerprint = examples_fingerprint(examples)
    logic_path = _logic_cache_path(cache_dir)

    print("[runtime] preparing neural runtime", flush=True)
    scorer = prepare_core(cache_dir, config.threshold, config.use_forgetting)
    try:
        print("[logic] preparing sentence-level logic cache", flush=True)
        logic_data = logic_for_examples(
            examples,
            cache_path=logic_path,
            batch_size=config.batch_size,
            roberta_batch_size=config.roberta_batch_size,
            reuse_logic_cache=reuse_logic_cache,
        )
        print("[logic] logic data ready", flush=True)
        reasoner.set_threshold(config.threshold)
        reasoner.set_use_forgetting(config.use_forgetting)
        results = run_prover(examples, logic_data)
    finally:
        scorer.save()

    payload = {
        "config": asdict(config),
        "examples_fingerprint": fingerprint,
        **results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.slug()}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[result] saved {output_path}", flush=True)
    return payload


def run_parameter_analysis(
    base_config: ExperimentConfig,
    thresholds: Iterable[float],
    data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    reuse_logic_cache: bool = True,
) -> list[dict]:
    """Run the paper's threshold sweep while reusing logic and neural caches."""
    ensure_nltk_data()
    examples = load_dataset_examples(
        base_config.dataset,
        data_dir=data_dir,
        per_class=base_config.per_class,
        seed=base_config.seed,
        max_len=base_config.max_len,
        include_esnli_explanation=base_config.include_explanation,
    )
    fingerprint = examples_fingerprint(examples)
    logic_path = _logic_cache_path(cache_dir)

    print("[runtime] preparing neural runtime", flush=True)
    scorer = prepare_core(cache_dir, base_config.threshold, base_config.use_forgetting)
    try:
        print("[logic] preparing sentence-level logic cache", flush=True)
        logic_data = logic_for_examples(
            examples,
            cache_path=logic_path,
            batch_size=base_config.batch_size,
            roberta_batch_size=base_config.roberta_batch_size,
            reuse_logic_cache=reuse_logic_cache,
        )
        print("[logic] logic data ready", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for threshold in thresholds:
            config = ExperimentConfig(
                dataset=base_config.dataset,
                per_class=base_config.per_class,
                seed=base_config.seed,
                max_len=base_config.max_len,
                threshold=threshold,
                batch_size=base_config.batch_size,
                roberta_batch_size=base_config.roberta_batch_size,
                use_forgetting=base_config.use_forgetting,
                include_explanation=base_config.include_explanation,
            )
            print(f"\n[analysis] threshold={threshold}", flush=True)
            reasoner.set_threshold(threshold)
            reasoner.set_use_forgetting(config.use_forgetting)
            result = run_prover(examples, logic_data)
            payload = {
                "config": asdict(config),
                "examples_fingerprint": fingerprint,
                **result,
            }
            output_path = output_dir / f"{config.slug()}.json"
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[result] saved {output_path}", flush=True)
            summaries.append(
                {
                    "threshold": threshold,
                    "accuracy": result["accuracy"],
                    "report": result["classification_report"],
                    "skipped_both": result["skipped_both"],
                    "errors": len(result["errors"]),
                }
            )
    finally:
        scorer.save()

    summary_path = output_dir / (
        f"parameter_analysis_{base_config.dataset}_seed{base_config.seed}"
        f"_n{base_config.per_class}_maxlen{base_config.max_len}"
        f"{base_config.mode_suffix()}.json"
    )
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[analysis] saved {summary_path}", flush=True)
    return summaries
