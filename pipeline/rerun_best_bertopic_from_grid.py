#!/usr/bin/env python3
"""
rerun_best_bertopic_from_grid.py

Bridge script for running the selected best grid-search configuration with the
existing pipeline/run_bertopic_from_manifest.py script.

Why this wrapper exists
-----------------------
The current run_bertopic_from_manifest.py always expects a manifest and uses
<output-dir>/chunks_input.parquet as its accumulated chunk file. To reuse one
shared top-level chunk file without rebuilding, this wrapper copies or symlinks
that common chunks_input.parquet into the final output directory and calls the
main runner with:

    --chunk-episode-limit 0 --train --force-train

Because chunk_episode_limit=0, the main runner loads the existing chunk file and
skips new chunk construction.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run best grid-search config with run_bertopic_from_manifest.py")
    p.add_argument("--best-config", required=True, help="Path to best_config.json from grid search.")
    p.add_argument("--manifest", required=True, help="Path to manifest.parquet for the main runner.")
    p.add_argument("--main-script", required=True, help="Path to pipeline/run_bertopic_from_manifest.py.")
    p.add_argument("--output-dir", required=True, help="Final BERTopic output directory for the best run.")
    p.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to place common chunks into output-dir. Use copy on filesystems where symlinks are restricted.",
    )
    p.add_argument("--execute", action="store_true", help="Actually run the command. Default only prints it.")
    p.add_argument("--python", default="python", help="Python executable to use.")
    return p.parse_args()


def add_bool_flag(cmd: List[str], flag: str, value: bool, *, supports_negative: bool = True) -> None:
    if value:
        cmd.append(flag)
    elif supports_negative:
        cmd.append("--no-" + flag[2:])


def add_value(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def command_from_config(args: argparse.Namespace, payload: Dict[str, Any]) -> List[str]:
    params = payload["run_bertopic_from_manifest_params"]
    cmd: List[str] = [
        args.python,
        str(Path(args.main_script).expanduser().resolve()),
        "--manifest",
        str(Path(args.manifest).expanduser().resolve()),
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--chunk-episode-limit",
        "0",
        "--train",
        "--force-train",
    ]

    value_flags = {
        "embedding_model": "--embedding-model",
        "embedding_device": "--embedding-device",
        "embedding_revision": "--embedding-revision",
        "stopwords": "--stopwords",
        "extra_stopwords": "--extra-stopwords",
        "name_stopwords_file": "--name-stopwords-file",
        "names_dataset_mode": "--names-dataset-mode",
        "names_dataset_countries": "--names-dataset-countries",
        "names_dataset_top_n": "--names-dataset-top-n",
        "names_dataset_min_country_prob": "--names-dataset-min-country-prob",
        "umap_n_neighbors": "--umap-n-neighbors",
        "umap_n_components": "--umap-n-components",
        "umap_min_dist": "--umap-min-dist",
        "umap_metric": "--umap-metric",
        "umap_random_state": "--umap-random-state",
        "hdbscan_min_cluster_size": "--hdbscan-min-cluster-size",
        "hdbscan_min_samples": "--hdbscan-min-samples",
        "hdbscan_metric": "--hdbscan-metric",
        "hdbscan_cluster_selection_method": "--hdbscan-cluster-selection-method",
        "vectorizer_ngram_min": "--vectorizer-ngram-min",
        "vectorizer_ngram_max": "--vectorizer-ngram-max",
        "vectorizer_min_df": "--vectorizer-min-df",
        "vectorizer_max_df": "--vectorizer-max-df",
        "nr_topics": "--nr-topics",
    }
    for key, flag in value_flags.items():
        add_value(cmd, flag, params.get(key))

    add_bool_flag(cmd, "--use-names-dataset", bool(params.get("use_names_dataset", True)))
    add_bool_flag(cmd, "--calculate-probabilities", bool(params.get("calculate_probabilities", False)))
    if params.get("save_probs", False):
        cmd.append("--save-probs")
    return cmd


def place_common_chunks(payload: Dict[str, Any], output_dir: Path, mode: str) -> Path:
    chunks_path = Path(payload["chunks_path"]).expanduser().resolve()
    if not chunks_path.exists():
        raise FileNotFoundError(f"Common chunks file from best_config.json not found: {chunks_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "chunks_input.parquet"
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(chunks_path)
    else:
        shutil.copy2(chunks_path, dst)

    return dst


def main() -> None:
    args = parse_args()
    best_config_path = Path(args.best_config).expanduser().resolve()
    payload = json.loads(best_config_path.read_text(encoding="utf-8"))

    output_dir = Path(args.output_dir).expanduser().resolve()
    placed = place_common_chunks(payload, output_dir, args.copy_mode)
    cmd = command_from_config(args, payload)

    print(f"Common chunks placed at: {placed}")
    print("Command:")
    print(" ".join(shlex.quote(x) for x in cmd))

    if args.execute:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
