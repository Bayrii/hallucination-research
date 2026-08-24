"""
Stage 1 — Embed the corpus and provide retrieval.

Embeds every abstract once with sentence-transformers/all-MiniLM-L6-v2 and
persists the vectors, so no later stage re-embeds the corpus.

This module is BOTH a script and a library. run_pipeline.py imports CorpusIndex
rather than shelling out.

VRAM / RAM
----------
all-MiniLM-L6-v2 is ~90 MB on disk, ~400 MB resident. It runs fine on CPU and
takes ~10s for 50 abstracts, so `--device cpu` is a perfectly good default even
on the 3060 box — worth doing while the generator holds VRAM.

Usage:
    python index_corpus.py                                    # build the index
    python index_corpus.py --query "how is faithfulness measured?" --k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    DATA_DIR,
    Timer,
    die,
    info,
    log_run_config,
    normalize_arxiv_id,
    require_file,
    rule,
    set_seed,
    truncate,
    write_json,
)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class CorpusIndex:
    """
    A loaded corpus + its embedding matrix, with cosine retrieval.

    The embedding model is loaded LAZILY, on the first call that needs to embed
    a query. Retrieval by ID (get_by_id) and forced-context lookups therefore
    cost no model load at all — which matters in run_pipeline.py, where the
    generator is already holding most of a 6 GB card.
    """

    def __init__(
        self,
        embeddings: Any,
        records: list[dict],
        model_name: str,
        device: str = "cpu",
    ) -> None:
        self.embeddings = embeddings  # (N, D) float32, L2-normalized
        self.records = records
        self.model_name = model_name
        self.device = device
        self._model = None
        self._id_to_pos = {
            normalize_arxiv_id(r["arxiv_id"]): i for i, r in enumerate(records)
        }

    # --- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        corpus_path: str | Path,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> "CorpusIndex":
        from common import read_json

        records = read_json(corpus_path)
        if not isinstance(records, list) or not records:
            die(f"{corpus_path} is not a non-empty JSON list.")

        missing = [
            i for i, r in enumerate(records) if not r.get("abstract") or not r.get("arxiv_id")
        ]
        if missing:
            die(
                f"{len(missing)} corpus entries lack 'abstract' or 'arxiv_id' "
                f"(first at index {missing[0]}).",
                "Fix or delete those entries in corpus.json and re-run.",
            )

        model = _load_model(model_name, device)
        texts = [_doc_text(r) for r in records]

        info(f"embedding {len(texts)} abstracts on {device} …")
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine reduces to a dot product
            show_progress_bar=True,
        )

        idx = cls(emb.astype("float32"), records, model_name, device)
        idx._model = model
        return idx

    @classmethod
    def load(cls, index_dir: str | Path = DATA_DIR, device: str = "cpu") -> "CorpusIndex":
        import numpy as np

        from common import read_json

        d = Path(index_dir)
        emb_path = require_file(d / "embeddings.npy", "python index_corpus.py")
        meta_path = require_file(d / "embeddings_meta.json", "python index_corpus.py")

        meta = read_json(meta_path)
        emb = np.load(emb_path)

        records = meta["records"]
        if emb.shape[0] != len(records):
            die(
                f"Index is inconsistent: {emb.shape[0]} vectors vs "
                f"{len(records)} records.",
                "The corpus changed after indexing. Re-run: python index_corpus.py",
            )
        return cls(emb, records, meta["model_name"], device)

    def save(self, out_dir: str | Path = DATA_DIR) -> None:
        import numpy as np

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.embeddings)
        write_json(
            d / "embeddings_meta.json",
            {
                "model_name": self.model_name,
                "dim": int(self.embeddings.shape[1]),
                "count": int(self.embeddings.shape[0]),
                "normalized": True,
                "records": self.records,
            },
        )

    # --- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name, self.device)
        return self._model

    def unload_model(self) -> None:
        """Free the encoder. Call before loading a generator on a 6 GB card."""
        import gc

        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def get_by_id(self, arxiv_id: str) -> dict | None:
        """Look up one record. No model load required."""
        pos = self._id_to_pos.get(normalize_arxiv_id(arxiv_id))
        return self.records[pos] if pos is not None else None

    def embed(self, texts: list[str]):
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Top-k abstracts by cosine similarity.

        Returns copies of the records with 'score' and 'rank' added, so callers
        can't accidentally mutate the index.
        """
        import numpy as np

        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, len(self.records))

        qv = self.embed([query])[0]
        scores = self.embeddings @ qv  # both sides L2-normalized -> cosine

        top = np.argsort(-scores)[:k]
        out = []
        for rank, i in enumerate(top):
            rec = dict(self.records[int(i)])
            rec["score"] = float(scores[int(i)])
            rec["rank"] = rank
            out.append(rec)
        return out

    def similarity_to_all(self, arxiv_id: str):
        """Cosine of one indexed abstract against every other. Used by the
        distractor picker in generate_qa_pairs.py — no query embedding needed,
        so no model load."""
        pos = self._id_to_pos.get(normalize_arxiv_id(arxiv_id))
        if pos is None:
            raise KeyError(f"unknown arxiv_id: {arxiv_id}")
        return self.embeddings @ self.embeddings[pos]


def _doc_text(record: dict) -> str:
    """
    Title + abstract as the embedded unit.

    Including the title measurably helps short queries like "what dataset?"
    latch onto the right paper; the abstract alone often buries the topic.
    """
    return f"{record.get('title', '').strip()}\n{record.get('abstract', '').strip()}"


def _load_model(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        die(
            "sentence-transformers is not installed.",
            "pip install -r requirements.txt",
        )

    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                info("CUDA requested but unavailable — falling back to CPU.")
                device = "cpu"
        except ImportError:
            device = "cpu"

    info(f"loading encoder {model_name} on {device}")
    return SentenceTransformer(model_name, device=device)


# --- convenience for interactive use ---------------------------------------

_DEFAULT_INDEX: CorpusIndex | None = None


def retrieve(query: str, k: int = 3, index_dir: str | Path = DATA_DIR) -> list[dict]:
    """
    Module-level retrieve(query, k) as specified, backed by a cached index.

    Convenient in a REPL. Inside a long-running script, prefer constructing a
    CorpusIndex explicitly so you control when the encoder is loaded and freed.
    """
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = CorpusIndex.load(index_dir)
    return _DEFAULT_INDEX.retrieve(query, k)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the corpus embedding index, or query it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--corpus", default=str(DATA_DIR / "corpus.json"))
    ap.add_argument("--index-dir", default=str(DATA_DIR))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--query",
        default=None,
        help="if given, load the existing index and retrieve instead of building",
    )
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    set_seed(args.seed)

    # --- query mode ---------------------------------------------------------
    if args.query:
        idx = CorpusIndex.load(args.index_dir, device=args.device)
        rule(f"Top-{args.k} for: {args.query!r}")
        with Timer() as t:
            hits = idx.retrieve(args.query, args.k)
        for h in hits:
            print(f"\n  [{h['rank']}] score={h['score']:.4f}  {h['arxiv_id']}")
            print(f"      {truncate(h['title'], 90)}")
            print(f"      {truncate(h['abstract'], 160)}")
        print(f"\n  ({t.ms:.0f} ms, index of {len(idx)} abstracts)\n")
        return

    # --- build mode ---------------------------------------------------------
    log_run_config("index_corpus", args)
    require_file(args.corpus, "python build_corpus.py")

    rule("Stage 1: index corpus")
    with Timer() as t:
        idx = CorpusIndex.build(
            args.corpus,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
        )
        idx.save(args.index_dir)

    rule("Done")
    info(
        f"{len(idx)} abstracts, dim={idx.embeddings.shape[1]} "
        f"-> {args.index_dir}/embeddings.npy   ({t.seconds:.1f}s)"
    )
    print(
        "\nSanity-check retrieval before moving on:\n\n"
        '    python index_corpus.py --query "how is faithfulness measured?" --k 5\n\n'
        "The top hits should be visibly on-topic. Then run:\n\n"
        "    python generate_qa_pairs.py\n"
    )


if __name__ == "__main__":
    main()
