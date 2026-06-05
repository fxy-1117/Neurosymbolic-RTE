"""Runtime loading and neural scoring utilities.

The notebook created the AMR parser and SentenceTransformer model as globals.
This module keeps those expensive objects behind explicit loader functions so
the command line entry points can cache results between runs.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict

import torch

warnings.filterwarnings(
    "ignore",
    message="An output with one or more elements was resized.*",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_nltk_data(nltk_data_dir: Path | None = None) -> None:
    """Prefer the local NLTK data directory when it exists."""
    nltk_data_dir = nltk_data_dir or project_root() / "nltk_data"
    if not nltk_data_dir.exists():
        return

    os.environ.setdefault("NLTK_DATA", str(nltk_data_dir))
    import nltk

    if str(nltk_data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_data_dir))


def prefer_local_fairseq_hub() -> None:
    """Use the local torch hub checkout of fairseq when it is already cached.

    transition-amr-parser calls torch.hub.load("pytorch/fairseq", "bart.large").
    Torch hub may try to contact the network even when the checkout is cached.
    For this notebook-to-Python runner, we redirect that one dependency to the
    local cache when available.
    """
    fairseq_cache = Path.home() / ".cache" / "torch" / "hub" / "pytorch_fairseq_main"
    if not fairseq_cache.exists():
        return

    original_load = torch.hub.load

    def local_first_load(source_or_dir, model, *args, **kwargs):
        if source_or_dir == "pytorch/fairseq":
            kwargs["source"] = "local"
            return original_load(str(fairseq_cache), model, *args, **kwargs)
        return original_load(source_or_dir, model, *args, **kwargs)

    torch.hub.load = local_first_load


def load_amr_runtime(model_name: str = "AMR3-joint-ontowiki-seed43"):
    """Load the AMR parser and AMR-to-logic converter used in the notebook."""
    prefer_local_fairseq_hub()
    from amr_logic_converter import AmrLogicConverter
    from transition_amr_parser.parse import AMRParser

    parser = AMRParser.from_pretrained(model_name)
    converter = AmrLogicConverter(
        existentially_quantify_instances=False,
        invert_relations=True,
    )
    return parser, converter


def load_spacy_model(name: str = "en_core_web_sm"):
    import spacy

    return spacy.load(name)


class NeuralScorer:
    """SentenceTransformer cosine scorer with persistent embedding cache."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_path: Path | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer, util

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.util = util
        self.cache_path = cache_path
        self.cache: Dict[str, torch.Tensor] = {}
        if cache_path and cache_path.exists():
            with cache_path.open("rb") as handle:
                self.cache = pickle.load(handle)

    def _embedding(self, text: str) -> torch.Tensor:
        text = "" if text is None else str(text)
        if text not in self.cache:
            embedding = self.model.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            self.cache[text] = embedding.detach().cpu()
        embedding = self.cache[text]
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)
            self.cache[text] = embedding
        return embedding

    def __call__(self, s1: str, s2: str):
        emb1 = self._embedding(s1)
        emb2 = self._embedding(s2)
        return self.util.pytorch_cos_sim(emb1, emb2)[0][0]

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as handle:
            pickle.dump(self.cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
