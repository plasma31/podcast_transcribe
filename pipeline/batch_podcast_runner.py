#!/usr/bin/env python3
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", required=True, help="Root folder containing podcast folders.")
    ap.add_argument("--out_root", required=True, help="Root folder for parquet/json outputs.")
    ap.add_argument("--state_dir", required=True, help="Folder for manifest and failure logs.")
    ap.add_argument("--whisper_model", default="small")
    ap.add_argument("--limit", type=int, default=500, help="How many episodes to process in this run.")
    ap.add_argument("--retry_failed", action="store_true", help="Include failed episodes in selection.")
    ap.add_argument("--rebuild_manifest", action="store_true", help="Rescan downloads root and refresh manifest.")
    ap.add_argument("--diar_gpu", action="store_true", help="Try to move pyannote to GPU.")
    ap.add_argument("--gender", action="store_true", help="Enable F0-based vocal gender estimation.")
    ap.add_argument("--max_speaker_sec", type=float, default=90.0)
    ap.add_argument("--min_turn_sec", type=float, default=1.0)
    ap.add_argument("--skip_existing_outputs", action="store_true", help="Mark episodes done if parquet outputs already exist.")
    return ap.parse_args()


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
            if episode_path.exists() and segments_path.exists():
                manifest.at[idx, "status"] = "done"
                manifest.at[idx, "output_episode_parquet"] = str(episode_path)
                manifest.at[idx, "output_segments_parquet"] = str(segments_path)
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
