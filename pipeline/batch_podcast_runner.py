#!/usr/bin/env python3
"""
batch_podcast_runner.py

Stage 2 resumable batch driver for the podcast corpus.

Purpose
-------
This script scans a local podcast-audio directory, maintains a persistent job
ledger, and processes selected episodes through `pipeline_core.PodcastPipeline`:

    audio file -> Whisper transcription -> pyannote diarization
    -> speaker matching -> optional F0 vocal-gender estimate
    -> episode/segment/debug artefacts

It is intentionally resumable. The manifest is saved before and after each
episode, and failed attempts are appended to a separate failure log. Re-running
the command does not duplicate known episodes because `episode_id` is derived
from the absolute audio path.

Typical command
---------------

    export PYANNOTE_TOKEN="<hugging-face-token>"

    python pipeline/batch_podcast_runner.py \
      --downloads /home/fdai7991/podcast_projekt/fyyd_downloads \
      --out_root /home/fdai7991/podcast_projekt/outputs \
      --state_dir /home/fdai7991/podcast_projekt/outputs/state \
      --whisper_model small \
      --limit 500 \
      --gender

Required inputs
---------------

--downloads
    Root directory containing one subdirectory per podcast. Audio files are
    discovered recursively below each podcast folder. Supported extensions are
    defined in `pipeline_core.AUDIO_EXTS`.

--out_root
    Root directory for generated Stage 2 artefacts. The runner writes:

        <out_root>/parquet/episodes/<episode_id>.parquet
        <out_root>/parquet/segments/<episode_id>.parquet
        <out_root>/json_debug/<episode_id>.json

--state_dir
    Directory for the resumability ledgers:

        <state_dir>/manifest.parquet
        <state_dir>/failures.parquet

Environment inputs
------------------

PYANNOTE_TOKEN, HF_TOKEN, or HUGGINGFACE_TOKEN
    Hugging Face access token for the gated pyannote diarization model. One of
    these variables must be set before running the script.

Main outputs
------------

manifest.parquet
    Current job ledger. Tracks inventory fields, status, attempt count, errors,
    runtime, and output paths.

failures.parquet
    Append-only failed-attempt log. A later successful retry updates the manifest
    but does not remove historical failure rows.

episodes/<episode_id>.parquet
    One row per processed episode with full transcript, speaker counts, runtime,
    Whisper language, and per-speaker gender JSON.

segments/<episode_id>.parquet
    One row per transcript segment with timing, speaker, text, and optional F0
    gender fields.

json_debug/<episode_id>.json
    Raw/debug payload containing Whisper segments, diarization turns, matched
    segments, and per-speaker F0 output.

Parameter notes
---------------

--rebuild_manifest
    Rescans the audio directory and merges it with the existing manifest. It does
    not delete Stage 2 artefacts. Existing episode statuses are preserved by
    `episode_id`.

--skip_existing_outputs
    Marks episodes as done when the expected episode and segment Parquet files
    already exist. This is useful when rebuilding or repairing the manifest after
    outputs were produced earlier.

--retry_failed
    Includes manifest rows with `status == failed` in the selected jobs. Without
    this flag, only `pending` and `running` rows are eligible. `running` rows are
    always retried so interrupted runs can recover.

--gender
    Enables the F0-based perceived vocal-gender estimate. Without this flag,
    segments are still transcribed and diarized, but gender fields default to
    `unknown` in downstream use.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import argparse
import gc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from pipeline_core import PodcastPipeline, build_episode_inventory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("podcast.batch")


MANIFEST_COLUMNS = [
    "episode_id",
    "podcast_folder",
    "podcast_dir",
    "episode_path",
    "episode_name",
    "audio_ext",
    "file_size_bytes",
    "mtime_ns",
    "status",
    "attempt_count",
    "last_error",
    "last_run_started_at",
    "last_run_finished_at",
    "runtime_sec",
    "output_episode_parquet",
    "output_segments_parquet",
    "output_debug_json",
]


class StateStore:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.state_dir / "manifest.parquet"
        self.failure_log_path = self.state_dir / "failures.parquet"

    def load_manifest(self) -> pd.DataFrame:
        if self.manifest_path.exists():
            df = pd.read_parquet(self.manifest_path)
            for col in MANIFEST_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[MANIFEST_COLUMNS]
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    def save_manifest(self, df: pd.DataFrame) -> None:
        ordered = df.copy()
        for col in MANIFEST_COLUMNS:
            if col not in ordered.columns:
                ordered[col] = None
        ordered = ordered[MANIFEST_COLUMNS].sort_values(["podcast_folder", "episode_path"]).reset_index(drop=True)
        ordered.to_parquet(self.manifest_path, index=False)

    def append_failure(self, row: Dict[str, Any]) -> None:
        if self.failure_log_path.exists():
            old = pd.read_parquet(self.failure_log_path)
            new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        else:
            new = pd.DataFrame([row])
        new.to_parquet(self.failure_log_path, index=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def merge_inventory_with_manifest(inventory_rows: List[Dict[str, Any]], existing: pd.DataFrame) -> pd.DataFrame:
    inventory = pd.DataFrame(inventory_rows)
    if inventory.empty:
        raise RuntimeError("No audio files found in downloads root.")

    if existing.empty:
        merged = inventory.copy()
        merged["status"] = "pending"
        merged["attempt_count"] = 0
        merged["last_error"] = None
        merged["last_run_started_at"] = None
        merged["last_run_finished_at"] = None
        merged["runtime_sec"] = None
        merged["output_episode_parquet"] = None
        merged["output_segments_parquet"] = None
        merged["output_debug_json"] = None
        return merged

    keep_cols = [c for c in MANIFEST_COLUMNS if c in existing.columns and c not in inventory.columns]
    existing_small = existing[["episode_id"] + keep_cols].drop_duplicates(subset=["episode_id"], keep="last")
    merged = inventory.merge(existing_small, on="episode_id", how="left")

    defaults = {
        "status": "pending",
        "attempt_count": 0,
        "last_error": None,
        "last_run_started_at": None,
        "last_run_finished_at": None,
        "runtime_sec": None,
        "output_episode_parquet": None,
        "output_segments_parquet": None,
        "output_debug_json": None,
    }
    for key, value in defaults.items():
        if key not in merged.columns:
            merged[key] = value
        else:
            merged[key] = merged[key].where(merged[key].notna(), value)

    return merged


def ensure_output_dirs(out_root: Path) -> Dict[str, Path]:
    paths = {
        "episodes": out_root / "parquet" / "episodes",
        "segments": out_root / "parquet" / "segments",
        "debug": out_root / "json_debug",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_artifacts(out_root: Path, artifacts) -> Dict[str, str]:
    paths = ensure_output_dirs(out_root)
    episode_id = artifacts.episode_record["episode_id"]

    episode_path = paths["episodes"] / f"{episode_id}.parquet"
    segments_path = paths["segments"] / f"{episode_id}.parquet"
    debug_path = paths["debug"] / f"{episode_id}.json"

    pd.DataFrame([artifacts.episode_record]).to_parquet(episode_path, index=False)
    pd.DataFrame(artifacts.segment_records).to_parquet(segments_path, index=False)
    debug_path.write_text(json.dumps(artifacts.debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "episode_parquet": str(episode_path),
        "segments_parquet": str(segments_path),
        "debug_json": str(debug_path),
    }


def select_jobs(df: pd.DataFrame, limit: int, retry_failed: bool, retry_running: bool = True) -> pd.DataFrame:
    allowed = {"pending"}
    if retry_failed:
        allowed.add("failed")
    if retry_running:
        allowed.add("running")
    jobs = df[df["status"].isin(allowed)].copy()
    jobs = jobs.sort_values(["podcast_folder", "episode_path"]).head(limit)
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 resumable batch runner: scan podcast audio, maintain "
            "manifest/failure ledgers, and write episode/segment/debug outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Outputs: <out_root>/parquet/episodes, <out_root>/parquet/segments, "
            "<out_root>/json_debug, plus <state_dir>/manifest.parquet and "
            "<state_dir>/failures.parquet. Requires PYANNOTE_TOKEN, HF_TOKEN, "
            "or HUGGINGFACE_TOKEN for pyannote."
        ),
    )

    required = parser.add_argument_group("required paths")
    required.add_argument(
        "--downloads",
        required=True,
        help=(
            "Root folder containing one subdirectory per podcast. Audio files are "
            "discovered recursively below each podcast folder."
        ),
    )
    required.add_argument(
        "--out_root",
        required=True,
        help=(
            "Root folder for generated Stage 2 artefacts. The runner writes "
            "parquet/episodes, parquet/segments, and json_debug below this path."
        ),
    )
    required.add_argument(
        "--state_dir",
        required=True,
        help=(
            "Folder for resumability ledgers. The runner writes manifest.parquet "
            "and failures.parquet here."
        ),
    )

    processing = parser.add_argument_group("processing controls")
    processing.add_argument(
        "--whisper_model",
        default="small",
        help=(
            "Whisper model size/name passed to whisper.load_model, for example "
            "tiny, base, small, medium, large, or a compatible local model name."
        ),
    )
    processing.add_argument(
        "--limit",
        type=int,
        default=500,
        help=(
            "Maximum number of eligible episodes to process in this run. Eligible "
            "means pending/running, plus failed when --retry_failed is set."
        ),
    )
    processing.add_argument(
        "--diar_gpu",
        action="store_true",
        help=(
            "Try to move the pyannote diarization pipeline to CUDA. Whisper uses "
            "CUDA automatically when torch.cuda.is_available() is true."
        ),
    )
    processing.add_argument(
        "--gender",
        action="store_true",
        help=(
            "Enable F0-based perceived vocal-gender estimation per diarized speaker. "
            "Without this flag, transcription and diarization still run, but gender "
            "fields default to unknown in downstream outputs."
        ),
    )
    processing.add_argument(
        "--max_speaker_sec",
        type=float,
        default=90.0,
        help=(
            "Maximum number of seconds of diarized speech concatenated per speaker "
            "for F0 estimation. Only relevant with --gender."
        ),
    )
    processing.add_argument(
        "--min_turn_sec",
        type=float,
        default=1.0,
        help=(
            "Minimum diarized turn duration, in seconds, included in the speaker F0 "
            "sample. Shorter turns are skipped. Only relevant with --gender."
        ),
    )

    resumability = parser.add_argument_group("resumability and retry controls")
    resumability.add_argument(
        "--retry_failed",
        action="store_true",
        help=(
            "Include rows with status=failed when selecting jobs. Historical failure "
            "rows remain in failures.parquet even if a retry later succeeds."
        ),
    )
    resumability.add_argument(
        "--rebuild_manifest",
        action="store_true",
        help=(
            "Rescan --downloads and merge the inventory with the existing manifest. "
            "This preserves known episode status by episode_id and does not delete outputs."
        ),
    )
    resumability.add_argument(
        "--skip_existing_outputs",
        action="store_true",
        help=(
            "Before selecting jobs, mark episodes as done when the expected episode "
            "and segment Parquet outputs already exist under --out_root. Useful after "
            "rebuilding or repairing the manifest."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hf_token = os.getenv("PYANNOTE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Missing Hugging Face token. Export PYANNOTE_TOKEN.")

    downloads_root = Path(args.downloads).resolve()
    out_root = Path(args.out_root).resolve()
    state_dir = Path(args.state_dir).resolve()

    store = StateStore(state_dir)
    existing = store.load_manifest()

    if args.rebuild_manifest or existing.empty:
        logger.info("Building or refreshing manifest from %s", downloads_root)
        inventory_rows = build_episode_inventory(str(downloads_root))
        manifest = merge_inventory_with_manifest(inventory_rows, existing)
        store.save_manifest(manifest)
    else:
        manifest = existing

    if args.skip_existing_outputs:
        logger.info("Checking for existing outputs to skip already finished episodes")
        for idx, row in manifest.iterrows():
            episode_path = out_root / "parquet" / "episodes" / f"{row['episode_id']}.parquet"
            segments_path = out_root / "parquet" / "segments" / f"{row['episode_id']}.parquet"
            debug_path = out_root / "json_debug" / f"{row['episode_id']}.json"
            if episode_path.exists() and segments_path.exists():
                manifest.at[idx, "status"] = "done"
                manifest.at[idx, "output_episode_parquet"] = str(episode_path)
                manifest.at[idx, "output_segments_parquet"] = str(segments_path)
                if debug_path.exists():
                    manifest.at[idx, "output_debug_json"] = str(debug_path)
        store.save_manifest(manifest)

    jobs = select_jobs(manifest, limit=args.limit, retry_failed=args.retry_failed)
    logger.info("Selected %s episode(s) for this run", len(jobs))
    if jobs.empty:
        return

    pipe = PodcastPipeline(
        whisper_model_size=args.whisper_model,
        hf_token=hf_token,
        use_gpu=True,
        diarization_on_gpu=args.diar_gpu,
        enable_gender=bool(args.gender),
        max_speaker_sec=args.max_speaker_sec,
        min_turn_sec=args.min_turn_sec,
    )

    manifest = store.load_manifest().set_index("episode_id")
    total = len(jobs)

    for count, (_, row) in enumerate(jobs.iterrows(), start=1):
        episode_id = row["episode_id"]
        logger.info("[%s/%s] Processing %s", count, total, row["episode_path"])

        manifest.at[episode_id, "status"] = "running"
        manifest.at[episode_id, "attempt_count"] = int(manifest.at[episode_id, "attempt_count"] or 0) + 1
        manifest.at[episode_id, "last_run_started_at"] = utc_now()
        manifest.at[episode_id, "last_error"] = None
        store.save_manifest(manifest.reset_index())

        try:
            artifacts = pipe.process_episode(row["episode_path"], podcast_folder=row["podcast_folder"])
            written = write_artifacts(out_root, artifacts)

            manifest.at[episode_id, "status"] = "done"
            manifest.at[episode_id, "runtime_sec"] = float(artifacts.episode_record["runtime_sec"])
            manifest.at[episode_id, "output_episode_parquet"] = written["episode_parquet"]
            manifest.at[episode_id, "output_segments_parquet"] = written["segments_parquet"]
            manifest.at[episode_id, "output_debug_json"] = written["debug_json"]
            manifest.at[episode_id, "last_run_finished_at"] = utc_now()
            logger.info("[%s/%s] Done: %s", count, total, episode_id)

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.exception("[%s/%s] Failed: %s", count, total, row["episode_path"])
            manifest.at[episode_id, "status"] = "failed"
            manifest.at[episode_id, "last_error"] = err
            manifest.at[episode_id, "last_run_finished_at"] = utc_now()
            store.append_failure(
                {
                    "episode_id": episode_id,
                    "episode_path": row["episode_path"],
                    "podcast_folder": row["podcast_folder"],
                    "attempt_count": manifest.at[episode_id, "attempt_count"],
                    "error": err,
                    "logged_at": utc_now(),
                }
            )
        finally:
            store.save_manifest(manifest.reset_index())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
