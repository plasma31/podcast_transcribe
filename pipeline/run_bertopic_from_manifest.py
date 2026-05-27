"""
Resumable BERTopic runner for the podcast pipeline.

Place this file in the same folder as:
  - bertopic_typisierung.py

It reads:
  - manifest.parquet
  - output_episode_parquet paths from the manifest
  - output_segments_parquet paths from the manifest

It builds/reuses chunk-level input with:

  chunk_id
  episode_id
  podcast_folder
  speaker
  gender
  start
  end
  chunk_text

Then it runs unsupervised BERTopic and saves:
  - chunks_input.parquet
  - chunk_build_state.parquet
  - chunks_with_topics.parquet
  - doc_topics.parquet / .csv
  - topic_info.parquet / .csv
  - topic_words.parquet / .csv
  - representative_docs.parquet / .csv
  - trained BERTopic model

Resumability:
  1. Chunk-building is episode-resumable.
     Already chunked episode_ids in chunk_build_state.parquet are skipped.
  2. BERTopic training is corpus-level.
     If a completed model marker exists, the script skips retraining unless --force-train is used.
  3. You may run chunk building in daily increments with --chunk-episode-limit.
  4. You may run training after enough chunks have accumulated with --train.

Typical usage:

  # Daily chunk build, no training yet
  python run_bertopic_from_manifest.py \
    --manifest /home/fdai7991/podcast_projekt/outputs/state/manifest.parquet \
    --output-dir /home/fdai7991/podcast_projekt/outputs/bertopic \
    --chunk-episode-limit 600 \
    --no-train

  # Train BERTopic on all accumulated chunks
  python run_bertopic_from_manifest.py \
    --manifest /home/fdai7991/podcast_projekt/outputs/state/manifest.parquet \
    --output-dir /home/fdai7991/podcast_projekt/outputs/bertopic \
    --train

  # Force retrain after changing parameters
  python run_bertopic_from_manifest.py \
    --manifest /home/fdai7991/podcast_projekt/outputs/state/manifest.parquet \
    --output-dir /home/fdai7991/podcast_projekt/outputs/bertopic \
    --train \
    --force-train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    from bertopic_typisierung import (
        _load_stopwords_from_file,
        _parse_country_list,
        _parse_extra_stopwords,
        build_model,
        build_stopwords,
        get_name_stopwords_from_names_dataset_all,
        get_name_stopwords_from_names_dataset_top,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not import helpers from bertopic_typisierung.py. "
        "Put this runner in the same pipeline folder as bertopic_typisierung.py."
    ) from exc


DEFAULT_NAMES_DATASET_COUNTRIES = "DE,AT,CH,TR,PL,RO,RU,UA,FR,IT,ES,PT,NL,BE,GB,US"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("podcast-bertopic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable unsupervised BERTopic runner for podcast parquet outputs."
    )

    parser.add_argument("--manifest", required=True, help="Path to manifest.parquet.")
    parser.add_argument("--output-dir", required=True, help="Output directory for BERTopic artifacts.")

    parser.add_argument(
        "--status",
        default="done",
        help="Manifest status to include. Default: done. Use 'all' for all rows.",
    )

    parser.add_argument(
        "--chunk-episode-limit",
        type=int,
        default=None,
        help="Maximum number of not-yet-chunked episodes to chunk in this run.",
    )

    parser.add_argument(
        "--rebuild-chunks",
        action="store_true",
        help="Delete existing chunk input/state and rebuild chunks from scratch.",
    )

    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run BERTopic after chunk building. Use --no-train for daily chunk-only runs.",
    )

    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain BERTopic even if a completed model marker exists.",
    )

    # Chunking
    parser.add_argument("--chunk-target-words", type=int, default=220)
    parser.add_argument("--chunk-min-words", type=int, default=80)
    parser.add_argument("--chunk-max-words", type=int, default=320)
    parser.add_argument("--min-segment-words", type=int, default=2)
    parser.add_argument("--min-doc-words", type=int, default=20)
    parser.add_argument(
        "--speaker-consistent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep chunks speaker-consistent where possible. Default: true.",
    )

    # Embeddings
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name/path.",
    )
    parser.add_argument(
        "--embedding-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Device for SentenceTransformer embeddings. "
            "Use 'cpu' if CUDA fails with no kernel image available. Default: auto."
        ),
    )
    parser.add_argument(
    "--embedding-revision",
    default=None,
    help="Optional Hugging Face model revision/branch/PR ref, e.g. refs/pr/116.",
)

    # Stopwords
    parser.add_argument("--stopwords", choices=["none", "de"], default="de")
    parser.add_argument("--extra-stopwords", type=str, default=None)
    parser.add_argument("--name-stopwords-file", type=str, default=None)
    parser.add_argument("--use-names-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--names-dataset-mode", choices=["top", "all"], default="top")
    parser.add_argument("--names-dataset-countries", type=str, default=DEFAULT_NAMES_DATASET_COUNTRIES)
    parser.add_argument("--names-dataset-top-n", type=int, default=20000)
    parser.add_argument("--names-dataset-min-country-prob", type=float, default=0.0)

    # UMAP
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-n-components", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--umap-metric", type=str, default="cosine")
    parser.add_argument("--umap-random-state", type=int, default=42)

    # HDBSCAN
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=50)
    parser.add_argument("--hdbscan-min-samples", type=int, default=1)
    parser.add_argument("--hdbscan-metric", type=str, default="euclidean")
    parser.add_argument("--hdbscan-cluster-selection-method", type=str, default="eom")

    # Vectorizer
    parser.add_argument("--vectorizer-ngram-min", type=int, default=1)
    parser.add_argument("--vectorizer-ngram-max", type=int, default=3)
    parser.add_argument("--vectorizer-min-df", type=int, default=5)
    parser.add_argument("--vectorizer-max-df", type=float, default=0.95)

    # Topic handling
    parser.add_argument("--nr-topics", type=int, default=None)
    parser.add_argument(
        "--calculate-probabilities",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default false for large corpora.",
    )
    parser.add_argument(
        "--save-probs",
        action="store_true",
        help="Save doc-topic probabilities. Can be very large.",
    )

    # Model save
    parser.add_argument("--model-dir-name", default="bertopic_model")
    parser.add_argument("--serialization", choices=["safetensors", "pytorch", "pickle"], default="safetensors")

    return parser.parse_args()


def now_iso() -> str:
    return pd.Timestamp.now('UTC').isoformat()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\\r", " ").replace("\\n", " ").replace("\\t", " ")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def word_count(text: str) -> int:
    return len(clean_text(text).split())


def stable_chunk_id(episode_id: str, start: Optional[float], end: Optional[float], idx: int, text: str) -> str:
    payload = f"{episode_id}|{start or 0:.3f}|{end or 0:.3f}|{idx}|{text[:200]}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def read_parquet_or_none(path: object, kind: str) -> Optional[pd.DataFrame]:
    if path is None or pd.isna(path):
        return None
    p = Path(str(path))
    if not p.exists():
        logger.warning("%s parquet missing: %s", kind, p)
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:
        logger.warning("Could not read %s parquet %s: %s", kind, p, exc)
        return None


def load_manifest(manifest_path: Path, status: str) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_parquet(manifest_path)

    required = {"episode_id", "output_episode_parquet", "output_segments_parquet"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    if status.lower() != "all" and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == status.lower()].copy()

    df = df[
        df["output_episode_parquet"].notna()
        & df["output_segments_parquet"].notna()
    ].copy()

    df = df.drop_duplicates(subset=["episode_id"], keep="last")
    logger.info("Manifest episodes available for chunking: %d", len(df))
    return df.reset_index(drop=True)


def state_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "chunks": output_dir / "chunks_input.parquet",
        "state": output_dir / "chunk_build_state.parquet",
        "failures": output_dir / "chunk_build_failures.parquet",
    }


def reset_chunk_state(output_dir: Path) -> None:
    paths = state_paths(output_dir)
    for p in paths.values():
        if p.exists():
            p.unlink()
            logger.info("Deleted %s", p)


def load_chunk_state(output_dir: Path) -> pd.DataFrame:
    p = state_paths(output_dir)["state"]
    if p.exists():
        return pd.read_parquet(p)

    return pd.DataFrame(
        columns=[
            "episode_id",
            "status",
            "n_chunks",
            "n_segments",
            "error",
            "processed_at",
            "output_episode_parquet",
            "output_segments_parquet",
        ]
    )


def save_chunk_state(output_dir: Path, state: pd.DataFrame) -> None:
    state_paths(output_dir)["state"].parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(state_paths(output_dir)["state"], index=False)


def append_chunk_failures(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    p = state_paths(output_dir)["failures"]
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_parquet(p)
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(p, index=False)


def load_existing_chunks(output_dir: Path) -> pd.DataFrame:
    p = state_paths(output_dir)["chunks"]
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(
        columns=[
            "chunk_id",
            "episode_id",
            "podcast_folder",
            "episode_path",
            "speaker",
            "gender",
            "start",
            "end",
            "chunk_text",
            "word_count",
            "source_segment_count",
        ]
    )


def save_chunks(output_dir: Path, chunks: pd.DataFrame) -> None:
    chunks = chunks.drop_duplicates(subset=["chunk_id"], keep="last").reset_index(drop=True)
    chunks.to_parquet(state_paths(output_dir)["chunks"], index=False)
    chunks.to_csv(output_dir / "chunks_input.csv", index=False)
    logger.info("Saved accumulated chunks: %d", len(chunks))


def join_episode_and_segments(row: pd.Series) -> pd.DataFrame:
    episode_id = str(row["episode_id"])

    seg_df = read_parquet_or_none(row.get("output_segments_parquet"), "segments")
    if seg_df is None or seg_df.empty:
        raise RuntimeError("segments parquet empty or unreadable")

    ep_df = read_parquet_or_none(row.get("output_episode_parquet"), "episode")

    seg_df = seg_df.copy()

    if "episode_id" not in seg_df.columns:
        seg_df["episode_id"] = episode_id
    if "podcast_folder" not in seg_df.columns:
        seg_df["podcast_folder"] = row.get("podcast_folder", None)
    if "episode_path" not in seg_df.columns:
        seg_df["episode_path"] = row.get("episode_path", None)

    # Join episode-level metadata where possible.
    if ep_df is not None and not ep_df.empty:
        ep_df = ep_df.copy()
        if "episode_id" not in ep_df.columns:
            ep_df["episode_id"] = episode_id

        metadata_cols = [c for c in ep_df.columns if c == "episode_id" or c not in seg_df.columns]
        ep_meta = ep_df[metadata_cols].drop_duplicates(subset=["episode_id"]).head(1)
        try:
            seg_df = seg_df.merge(ep_meta, on="episode_id", how="left")
        except Exception as exc:
            logger.warning("Episode metadata join failed for %s: %s", episode_id, exc)

    for col in ["speaker", "gender"]:
        if col not in seg_df.columns:
            seg_df[col] = "unknown"

    if "start" not in seg_df.columns:
        seg_df["start"] = None
    if "end" not in seg_df.columns:
        seg_df["end"] = None
    if "text" not in seg_df.columns:
        raise ValueError("segments parquet does not contain a text column")

    seg_df["text"] = seg_df["text"].map(clean_text)
    seg_df = seg_df[seg_df["text"].str.len() > 0].copy()
    return seg_df


def flush_chunk(rows: List[pd.Series], chunk_idx: int, chunks: List[Dict[str, object]]) -> int:
    if not rows:
        return chunk_idx

    texts = [clean_text(r.get("text", "")) for r in rows]
    chunk_text = " ".join([t for t in texts if t]).strip()
    if not chunk_text:
        return chunk_idx

    first = rows[0]
    last = rows[-1]

    episode_id = str(first.get("episode_id"))
    podcast_folder = first.get("podcast_folder")
    episode_path = first.get("episode_path")

    speakers = sorted(
        {
            str(r.get("speaker", "unknown"))
            for r in rows
            if str(r.get("speaker", "")).strip()
        }
    )
    genders = sorted(
        {
            str(r.get("gender", "unknown"))
            for r in rows
            if str(r.get("gender", "")).strip()
        }
    )

    speaker = speakers[0] if len(speakers) == 1 else "mixed"
    gender = genders[0] if len(genders) == 1 else "mixed"

    try:
        start = float(first.get("start")) if pd.notna(first.get("start")) else None
    except Exception:
        start = None
    try:
        end = float(last.get("end")) if pd.notna(last.get("end")) else None
    except Exception:
        end = None

    chunks.append(
        {
            "chunk_id": stable_chunk_id(episode_id, start, end, chunk_idx, chunk_text),
            "episode_id": episode_id,
            "podcast_folder": podcast_folder,
            "episode_path": episode_path,
            "speaker": speaker,
            "gender": gender,
            "start": start,
            "end": end,
            "chunk_text": chunk_text,
            "word_count": word_count(chunk_text),
            "source_segment_count": len(rows),
        }
    )
    return chunk_idx + 1


def build_chunks_for_episode(
    seg_df: pd.DataFrame,
    *,
    target_words: int,
    max_words: int,
    min_segment_words: int,
    min_doc_words: int,
    speaker_consistent: bool,
) -> pd.DataFrame:
    sort_cols = [c for c in ["episode_id", "start", "end", "segment_idx"] if c in seg_df.columns]
    if sort_cols:
        seg_df = seg_df.sort_values(sort_cols).copy()

    seg_df = seg_df.copy()
    seg_df["text"] = seg_df["text"].map(clean_text)
    seg_df["segment_word_count"] = seg_df["text"].map(word_count)
    seg_df = seg_df[seg_df["segment_word_count"] >= int(min_segment_words)].copy()

    chunks: List[Dict[str, object]] = []
    current_rows: List[pd.Series] = []
    current_words = 0
    current_speaker: Optional[str] = None
    chunk_idx = 0

    for _, seg in seg_df.iterrows():
        seg_words = int(seg["segment_word_count"])
        seg_speaker = str(seg.get("speaker", "unknown"))

        force_flush = False
        if current_rows:
            if speaker_consistent and current_speaker is not None and seg_speaker != current_speaker:
                force_flush = True
            if current_words >= target_words:
                force_flush = True
            if current_words + seg_words > max_words:
                force_flush = True

        if force_flush:
            chunk_idx = flush_chunk(current_rows, chunk_idx, chunks)
            current_rows = []
            current_words = 0
            current_speaker = None

        if not current_rows:
            current_speaker = seg_speaker

        current_rows.append(seg)
        current_words += seg_words

    if current_rows:
        chunk_idx = flush_chunk(current_rows, chunk_idx, chunks)

    out = pd.DataFrame(chunks)
    if out.empty:
        return out

    out = out[out["word_count"] >= int(min_doc_words)].copy()
    return out.reset_index(drop=True)


def build_chunks_resumable(manifest: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_chunks:
        reset_chunk_state(output_dir)

    existing_chunks = load_existing_chunks(output_dir)
    state = load_chunk_state(output_dir)

    done_ids = set(
        state.loc[state["status"].astype(str) == "done", "episode_id"].astype(str).tolist()
        if not state.empty else []
    )

    todo = manifest[~manifest["episode_id"].astype(str).isin(done_ids)].copy()

    if args.chunk_episode_limit is not None:
        todo = todo.head(int(args.chunk_episode_limit)).copy()

    logger.info("Episodes already chunked: %d", len(done_ids))
    logger.info("Episodes selected for chunking in this run: %d", len(todo))

    new_chunk_frames: List[pd.DataFrame] = []
    new_state_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []

    for i, row in todo.iterrows():
        episode_id = str(row["episode_id"])
        logger.info("[%d/%d] Chunking episode %s", len(new_state_rows) + len(failure_rows) + 1, len(todo), episode_id)
        t0 = time.perf_counter()

        try:
            seg_df = join_episode_and_segments(row)
            episode_chunks = build_chunks_for_episode(
                seg_df,
                target_words=args.chunk_target_words,
                max_words=args.chunk_max_words,
                min_segment_words=args.min_segment_words,
                min_doc_words=args.min_doc_words,
                speaker_consistent=bool(args.speaker_consistent),
            )

            if episode_chunks.empty:
                raise RuntimeError("no chunks created")

            new_chunk_frames.append(episode_chunks)
            new_state_rows.append(
                {
                    "episode_id": episode_id,
                    "status": "done",
                    "n_chunks": int(len(episode_chunks)),
                    "n_segments": int(len(seg_df)),
                    "error": None,
                    "processed_at": now_iso(),
                    "runtime_sec": float(time.perf_counter() - t0),
                    "output_episode_parquet": row.get("output_episode_parquet"),
                    "output_segments_parquet": row.get("output_segments_parquet"),
                }
            )

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.exception("Chunking failed for episode %s", episode_id)
            new_state_rows.append(
                {
                    "episode_id": episode_id,
                    "status": "failed",
                    "n_chunks": 0,
                    "n_segments": 0,
                    "error": err,
                    "processed_at": now_iso(),
                    "runtime_sec": float(time.perf_counter() - t0),
                    "output_episode_parquet": row.get("output_episode_parquet"),
                    "output_segments_parquet": row.get("output_segments_parquet"),
                }
            )
            failure_rows.append(
                {
                    "episode_id": episode_id,
                    "error": err,
                    "logged_at": now_iso(),
                    "output_episode_parquet": row.get("output_episode_parquet"),
                    "output_segments_parquet": row.get("output_segments_parquet"),
                }
            )

        # Periodic checkpoint every 25 episodes.
        if (len(new_state_rows) + len(failure_rows)) % 25 == 0:
            partial_state = pd.concat([state, pd.DataFrame(new_state_rows)], ignore_index=True)
            partial_state = partial_state.drop_duplicates(subset=["episode_id"], keep="last")
            save_chunk_state(output_dir, partial_state)

            partial_chunks = existing_chunks
            if new_chunk_frames:
                partial_chunks = pd.concat([existing_chunks] + new_chunk_frames, ignore_index=True)
            save_chunks(output_dir, partial_chunks)
            append_chunk_failures(output_dir, failure_rows)
            failure_rows = []

    if new_state_rows:
        state = pd.concat([state, pd.DataFrame(new_state_rows)], ignore_index=True)
        state = state.drop_duplicates(subset=["episode_id"], keep="last")
        save_chunk_state(output_dir, state)

    if failure_rows:
        append_chunk_failures(output_dir, failure_rows)

    if new_chunk_frames:
        all_chunks = pd.concat([existing_chunks] + new_chunk_frames, ignore_index=True)
    else:
        all_chunks = existing_chunks

    if all_chunks.empty:
        raise RuntimeError("No chunks available. Build chunks first or check manifest outputs.")

    save_chunks(output_dir, all_chunks)

    logger.info("Chunk build state summary:")
    if not state.empty:
        logger.info("\n%s", state["status"].value_counts(dropna=False).to_string())

    return all_chunks.reset_index(drop=True)


def build_stopword_list(args: argparse.Namespace) -> Tuple[Optional[List[str]], str]:
    extra_stopwords = _parse_extra_stopwords(args.extra_stopwords)
    extra_stopwords.extend(["müller", "özdal", "özsoy"])

    if args.use_names_dataset:
        countries = _parse_country_list(args.names_dataset_countries)
        before = len(extra_stopwords)

        if args.names_dataset_mode == "all":
            extra_stopwords.extend(
                get_name_stopwords_from_names_dataset_all(
                    countries=countries,
                    min_country_prob=float(args.names_dataset_min_country_prob),
                )
            )
        else:
            if countries is None:
                raise ValueError("--names-dataset-countries ALL is only supported with --names-dataset-mode all")
            extra_stopwords.extend(
                get_name_stopwords_from_names_dataset_top(
                    countries=countries,
                    top_n=int(args.names_dataset_top_n),
                )
            )

        logger.info("names-dataset stopwords added: %d", len(extra_stopwords) - before)

    extra_stopwords.extend(_load_stopwords_from_file(args.name_stopwords_file))

    if args.stopwords == "de":
        return build_stopwords(extra_stopwords), "sw-de"
    return None, "sw-none"


def topic_words_frame(model, topic_info: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for topic_id in topic_info["Topic"].tolist():
        words = model.get_topic(topic_id)
        if not words:
            continue
        rows.append(
            {
                "topic": int(topic_id),
                "words": ", ".join([w for w, _ in words]),
                "word_scores_json": json.dumps(
                    [{"word": str(w), "score": float(score)} for w, score in words],
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(rows)


def representative_docs_frame(model) -> pd.DataFrame:
    try:
        rep = model.get_representative_docs()
    except Exception:
        return pd.DataFrame(columns=["topic", "representative_doc_rank", "representative_doc"])

    rows: List[Dict[str, object]] = []
    if isinstance(rep, dict):
        for topic, docs in rep.items():
            for rank, doc in enumerate(docs or []):
                rows.append(
                    {
                        "topic": int(topic),
                        "representative_doc_rank": int(rank),
                        "representative_doc": str(doc),
                    }
                )
    return pd.DataFrame(rows)


def training_complete_marker(run_dir: Path) -> Path:
    return run_dir / "_TRAINING_COMPLETE.json"


def run_dir_for(args: argparse.Namespace, output_dir: Path) -> Path:
    tag = "sw-de" if args.stopwords == "de" else "sw-none"
    return output_dir / f"podcast_chunks_{tag}"


def run_bertopic(chunks: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    run_dir = run_dir_for(args, output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    marker = training_complete_marker(run_dir)
    model_dir = run_dir / args.model_dir_name

    if marker.exists() and model_dir.exists() and not args.force_train:
        logger.info("BERTopic model already completed at %s", run_dir)
        logger.info("Use --force-train to retrain.")
        return

    if "chunk_text" not in chunks.columns:
        raise ValueError("chunks dataframe must contain chunk_text")

    chunks = chunks.copy()
    chunks["chunk_text"] = chunks["chunk_text"].map(clean_text)
    chunks = chunks[chunks["chunk_text"].str.len() > 0].reset_index(drop=True)

    texts = chunks["chunk_text"].tolist()
    logger.info("BERTopic documents/chunks: %d", len(texts))

    stopwords, _tag = build_stopword_list(args)

    logger.info("Loading SentenceTransformer: %s", args.embedding_model)
    st_kwargs = {}
    if args.embedding_revision:
        st_kwargs["revision"] = args.embedding_revision

    if args.embedding_device == "auto":
        embedding_model = SentenceTransformer(args.embedding_model, **st_kwargs)
    else:
        embedding_model = SentenceTransformer(
            args.embedding_model,
            device=args.embedding_device,
            **st_kwargs,
        )

    logger.info("Building BERTopic model")
    model = build_model(
        stopwords=stopwords,
        embedding_model=embedding_model,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_n_components=args.umap_n_components,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap_random_state=args.umap_random_state,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_metric=args.hdbscan_metric,
        hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
        vectorizer_ngram_min=args.vectorizer_ngram_min,
        vectorizer_ngram_max=args.vectorizer_ngram_max,
        vectorizer_min_df=args.vectorizer_min_df,
        vectorizer_max_df=args.vectorizer_max_df,
    )

    model.calculate_probabilities = bool(args.calculate_probabilities or args.save_probs)

    logger.info("Fitting BERTopic")
    t0 = time.perf_counter()
    topics, probs = model.fit_transform(texts)

    if args.nr_topics is not None:
        logger.info("Reducing topics to %s", args.nr_topics)
        model.reduce_topics(texts, nr_topics=int(args.nr_topics))
        topics, probs = model.transform(texts)

    chunks_with_topics = chunks.copy()
    chunks_with_topics["doc_id"] = range(len(chunks_with_topics))
    chunks_with_topics["topic"] = topics

    logger.info("Saving topic-enriched chunk outputs")
    chunks_with_topics.to_parquet(run_dir / "chunks_with_topics.parquet", index=False)
    chunks_with_topics.to_csv(run_dir / "chunks_with_topics.csv", index=False)

    doc_cols = [
        "doc_id",
        "chunk_id",
        "episode_id",
        "podcast_folder",
        "episode_path",
        "speaker",
        "gender",
        "start",
        "end",
        "word_count",
        "source_segment_count",
        "topic",
        "chunk_text",
    ]
    doc_topics = chunks_with_topics[[c for c in doc_cols if c in chunks_with_topics.columns]].copy()
    doc_topics.to_parquet(run_dir / "doc_topics.parquet", index=False)
    doc_topics.to_csv(run_dir / "doc_topics.csv", index=False)

    topic_info = model.get_topic_info()
    topic_info.to_parquet(run_dir / "topic_info.parquet", index=False)
    topic_info.to_csv(run_dir / "topic_info.csv", index=False)

    topic_words = topic_words_frame(model, topic_info)
    topic_words.to_parquet(run_dir / "topic_words.parquet", index=False)
    topic_words.to_csv(run_dir / "topic_words.csv", index=False)

    representative_docs = representative_docs_frame(model)
    representative_docs.to_parquet(run_dir / "representative_docs.parquet", index=False)
    representative_docs.to_csv(run_dir / "representative_docs.csv", index=False)

    if args.save_probs and probs is not None:
        logger.info("Saving doc-topic probabilities")
        probs_df = pd.DataFrame(probs)
        probs_df.insert(0, "doc_id", range(len(probs_df)))
        probs_df.insert(1, "chunk_id", chunks_with_topics["chunk_id"].tolist())
        probs_df.to_parquet(run_dir / "doc_topic_probs.parquet", index=False)

    logger.info("Saving BERTopic model to %s", model_dir)
    try:
        model.save(model_dir, serialization=args.serialization, save_embedding_model=True)
    except TypeError:
        model.save(model_dir, serialization=args.serialization)

    logger.info("Saving visualizations")
    try:
        fig = model.visualize_topics()
        fig.write_html(run_dir / "topics_overview.html", include_plotlyjs="cdn")
    except Exception as exc:
        logger.warning("Could not save topics_overview.html: %s", exc)

    try:
        fig = model.visualize_barchart()
        fig.write_html(run_dir / "topics_barchart.html", include_plotlyjs="cdn")
    except Exception as exc:
        logger.warning("Could not save topics_barchart.html: %s", exc)

    try:
        fig = model.visualize_hierarchy()
        fig.write_html(run_dir / "topics_hierarchy.html", include_plotlyjs="cdn")
    except Exception as exc:
        logger.warning("Could not save topics_hierarchy.html: %s", exc)

    config = vars(args).copy()
    config.update(
        {
            "completed_at": now_iso(),
            "runtime_sec": float(time.perf_counter() - t0),
            "n_chunks": int(len(chunks_with_topics)),
            "n_topics_including_outlier": int(topic_info.shape[0]),
            "outlier_chunks": int((chunks_with_topics["topic"] == -1).sum()),
            "output_run_dir": str(run_dir),
            "model_dir": str(model_dir),
        }
    )
    (run_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("BERTopic complete: %s", run_dir)


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path, args.status)
    chunks = build_chunks_resumable(manifest, output_dir, args)

    if not args.train:
        logger.info("--no-train was set. Stopping after resumable chunk build.")
        return

    run_bertopic(chunks, output_dir, args)

if __name__ == "__main__":
    main()