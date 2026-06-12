#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def audit_manifest(manifest_path: Path, check_segments: bool = True) -> pd.DataFrame:
    manifest = pd.read_parquet(manifest_path)
    rows: List[Dict[str, Any]] = []

    for _, row in manifest.iterrows():
        issue_codes: List[str] = []
        issue_details: List[str] = []

        output_episode = row.get("output_episode_parquet")
        output_segments = row.get("output_segments_parquet")

        if _is_missing(output_episode):
            issue_codes.append("missing_output_episode_parquet")
            issue_details.append("Manifest row has no output_episode_parquet path.")
        else:
            episode_path = Path(str(output_episode))
            if not episode_path.exists():
                issue_codes.append("episode_parquet_not_found")
                issue_details.append(f"Episode parquet not found: {episode_path}")
            else:
                ep_df = pd.read_parquet(episode_path)
                if ep_df.empty:
                    issue_codes.append("episode_parquet_empty")
                    issue_details.append("Episode parquet exists but has no rows.")
                else:
                    ep = ep_df.iloc[0]

                    n_speakers = ep.get("n_speakers")
                    if _is_missing(n_speakers):
                        issue_codes.append("missing_n_speakers")
                        issue_details.append("Episode parquet missing n_speakers.")
                    else:
                        try:
                            if int(n_speakers) <= 0:
                                issue_codes.append("no_speakers_detected")
                                issue_details.append(f"n_speakers={n_speakers}")
                        except Exception:
                            issue_codes.append("invalid_n_speakers")
                            issue_details.append(f"Could not parse n_speakers={n_speakers!r}")

                    speakers = _safe_json_loads(ep.get("speakers_json"))
                    if speakers is None:
                        issue_codes.append("missing_speakers_json")
                        issue_details.append("speakers_json missing or unreadable.")
                    elif isinstance(speakers, list) and len(speakers) == 0:
                        issue_codes.append("empty_speakers_json")
                        issue_details.append("speakers_json is an empty list.")

                    speaker_gender = _safe_json_loads(ep.get("speaker_gender_json"))
                    if speaker_gender is None:
                        issue_codes.append("missing_speaker_gender_json")
                        issue_details.append("speaker_gender_json missing or unreadable.")
                    elif isinstance(speaker_gender, dict) and len(speaker_gender) == 0:
                        issue_codes.append("empty_speaker_gender_json")
                        issue_details.append("speaker_gender_json is an empty object.")
                    elif isinstance(speaker_gender, dict):
                        valid_gender_entries = 0
                        unknown_only = True
                        for spk, payload in speaker_gender.items():
                            if isinstance(payload, dict):
                                label = payload.get("label")
                                if label is not None:
                                    valid_gender_entries += 1
                                if label not in {"unknown", None, ""}:
                                    unknown_only = False
                        if valid_gender_entries == 0:
                            issue_codes.append("speaker_gender_without_labels")
                            issue_details.append("speaker_gender_json has no usable label entries.")
                        elif unknown_only:
                            issue_codes.append("speaker_gender_unknown_only")
                            issue_details.append("speaker_gender_json contains only unknown labels.")

        if check_segments and not _is_missing(output_segments):
            seg_path = Path(str(output_segments))
            if seg_path.exists():
                seg_df = pd.read_parquet(seg_path)
                if seg_df.empty:
                    issue_codes.append("segments_parquet_empty")
                    issue_details.append("Segments parquet exists but has no rows.")
                else:
                    if "speaker" not in seg_df.columns:
                        issue_codes.append("segments_missing_speaker_column")
                        issue_details.append("Segments parquet has no speaker column.")
                    else:
                        speaker_non_null = seg_df["speaker"].fillna("").astype(str).str.strip().ne("").sum()
                        if speaker_non_null == 0:
                            issue_codes.append("segments_without_speaker_values")
                            issue_details.append("Segments parquet speaker column is entirely empty.")

                    if "gender" not in seg_df.columns:
                        issue_codes.append("segments_missing_gender_column")
                        issue_details.append("Segments parquet has no gender column.")
                    else:
                        gender_series = seg_df["gender"].fillna("").astype(str).str.strip()
                        non_empty = gender_series.ne("").sum()
                        non_unknown = (~gender_series.isin(["", "unknown", "Unknown", "nan", "None"])) .sum()
                        if non_empty == 0:
                            issue_codes.append("segments_without_gender_values")
                            issue_details.append("Segments parquet gender column is entirely empty.")
                        elif non_unknown == 0:
                            issue_codes.append("segments_gender_unknown_only")
                            issue_details.append("All segment gender values are unknown.")
            else:
                issue_codes.append("segments_parquet_not_found")
                issue_details.append(f"Segments parquet not found: {seg_path}")

        if issue_codes:
            rows.append(
                {
                    "episode_id": row.get("episode_id"),
                    "status": row.get("status"),
                    "podcast_folder": row.get("podcast_folder"),
                    "episode_name": row.get("episode_name"),
                    "episode_path": row.get("episode_path"),
                    "output_episode_parquet": row.get("output_episode_parquet"),
                    "output_segments_parquet": row.get("output_segments_parquet"),
                    "issue_codes": "|".join(sorted(set(issue_codes))),
                    "issue_count": len(set(issue_codes)),
                    "issue_details": " ; ".join(issue_details),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find episodes in a manifest whose parquet outputs are missing speaker and/or gender analysis."
    )
    ap.add_argument("--manifest", required=True, help="Path to manifest.parquet")
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output path. Supports .csv or .parquet. Default: manifest sibling audit_missing_speaker_gender.parquet",
    )
    ap.add_argument(
        "--no_segments_check",
        action="store_true",
        help="Only inspect manifest + output_episode_parquet, skip output_segments_parquet validation.",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    result = audit_manifest(manifest_path, check_segments=not args.no_segments_check)

    if args.out is None:
        out_path = manifest_path.parent / "audit_missing_speaker_gender.parquet"
    else:
        out_path = Path(args.out).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        result.to_csv(out_path, index=False)
    else:
        result.to_parquet(out_path, index=False)

    total = len(pd.read_parquet(manifest_path))
    flagged = len(result)
    print(f"Manifest rows checked: {total}")
    print(f"Rows flagged: {flagged}")
    print(f"Output written to: {out_path}")

    if flagged:
        print("\nTop issue counts:")
        counts = (
            result["issue_codes"]
            .str.split("|", regex=False)
            .explode()
            .value_counts(dropna=False)
        )
        for issue, count in counts.head(20).items():
            print(f"  {issue}: {count}")


if __name__ == "__main__":
    main()
