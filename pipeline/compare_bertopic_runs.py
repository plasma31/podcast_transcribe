#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd


def find_bertopic_runs(root: Path):
    """
    Find folders that look like BERTopic run folders.
    Expected structure:
      <some_output_dir>/podcast_chunks_sw-de/doc_topics.parquet
      <some_output_dir>/podcast_chunks_sw-de/topic_info.parquet
      <some_output_dir>/podcast_chunks_sw-de/topic_words.parquet
    """
    runs = []

    for doc_path in root.rglob("doc_topics.parquet"):
        run_dir = doc_path.parent
        topic_info_path = run_dir / "topic_info.parquet"
        topic_words_path = run_dir / "topic_words.parquet"

        if topic_info_path.exists() and topic_words_path.exists():
            runs.append(run_dir)

    return sorted(set(runs))


def summarize_run(run_dir: Path):
    docs_path = run_dir / "doc_topics.parquet"
    info_path = run_dir / "topic_info.parquet"
    words_path = run_dir / "topic_words.parquet"

    docs = pd.read_parquet(docs_path)
    info = pd.read_parquet(info_path)
    words = pd.read_parquet(words_path)

    n_chunks = len(docs)
    n_episodes = docs["episode_id"].nunique() if "episode_id" in docs.columns else None
    n_podcasts = docs["podcast_folder"].nunique() if "podcast_folder" in docs.columns else None

    n_topics_including_outlier = len(info)
    n_topics_excluding_outlier = int((info["Topic"] != -1).sum()) if "Topic" in info.columns else None

    outlier_count = int((docs["topic"] == -1).sum()) if "topic" in docs.columns else None
    outlier_rate = outlier_count / n_chunks if n_chunks else None

    non_outlier_docs = docs[docs["topic"] != -1] if "topic" in docs.columns else docs
    assigned_count = len(non_outlier_docs)
    assigned_rate = assigned_count / n_chunks if n_chunks else None

    if "topic" in docs.columns:
        topic_counts = docs["topic"].value_counts()
        largest_topic = int(topic_counts.index[0])
        largest_topic_count = int(topic_counts.iloc[0])
        largest_non_outlier = docs[docs["topic"] != -1]["topic"].value_counts()
        if len(largest_non_outlier):
            largest_non_outlier_topic = int(largest_non_outlier.index[0])
            largest_non_outlier_count = int(largest_non_outlier.iloc[0])
        else:
            largest_non_outlier_topic = None
            largest_non_outlier_count = None
    else:
        largest_topic = None
        largest_topic_count = None
        largest_non_outlier_topic = None
        largest_non_outlier_count = None

    run_name = run_dir.parent.name
    variant = run_dir.name

    return {
        "run_name": run_name,
        "variant": variant,
        "run_dir": str(run_dir),
        "chunks": n_chunks,
        "episodes": n_episodes,
        "podcasts": n_podcasts,
        "topics_including_outlier": n_topics_including_outlier,
        "topics_excluding_outlier": n_topics_excluding_outlier,
        "outlier_count": outlier_count,
        "outlier_rate": outlier_rate,
        "assigned_count": assigned_count,
        "assigned_rate": assigned_rate,
        "largest_topic": largest_topic,
        "largest_topic_count": largest_topic_count,
        "largest_non_outlier_topic": largest_non_outlier_topic,
        "largest_non_outlier_count": largest_non_outlier_count,
    }


def get_top_topics(run_dir: Path, n: int):
    info = pd.read_parquet(run_dir / "topic_info.parquet")
    cols = [c for c in ["Topic", "Count", "Name"] if c in info.columns]
    return info[cols].head(n)


def get_top_words(run_dir: Path, n: int):
    words = pd.read_parquet(run_dir / "topic_words.parquet")
    cols = [c for c in ["topic", "words"] if c in words.columns]
    return words[cols].head(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/fdai7991/podcast_projekt/outputs",
        help="Root folder to search for BERTopic runs.",
    )
    parser.add_argument(
        "--out",
        default="/home/fdai7991/podcast_projekt/outputs/bertopic_run_comparison.csv",
        help="CSV output path for comparison table.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top topics/words to print per run.",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Only show runs with more than 100,000 chunks.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)

    run_dirs = find_bertopic_runs(root)

    if not run_dirs:
        print(f"No BERTopic runs found under: {root}")
        return

    summaries = []
    for run_dir in run_dirs:
        try:
            summary = summarize_run(run_dir)
            summaries.append(summary)
        except Exception as e:
            print(f"Skipping unreadable run: {run_dir}")
            print(f"  Error: {type(e).__name__}: {e}")

    if not summaries:
        print("No readable BERTopic runs found.")
        return

    summary_df = pd.DataFrame(summaries)

    if args.full_only:
        summary_df = summary_df[summary_df["chunks"] > 100_000].copy()

    summary_df = summary_df.sort_values(
        ["chunks", "outlier_rate", "topics_excluding_outlier"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n=== BERTopic run comparison ===")
    display_cols = [
        "run_name",
        "variant",
        "chunks",
        "episodes",
        "podcasts",
        "topics_including_outlier",
        "topics_excluding_outlier",
        "outlier_rate",
        "assigned_rate",
        "largest_non_outlier_topic",
        "largest_non_outlier_count",
        "run_dir",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved comparison CSV to: {out_path}")

    print("\n=== Top topics per run ===")
    for _, row in summary_df.iterrows():
        run_dir = Path(row["run_dir"])
        print("\n" + "=" * 120)
        print(f"RUN: {row['run_name']} / {row['variant']}")
        print(f"DIR: {run_dir}")
        print(f"chunks={row['chunks']} | topics={row['topics_including_outlier']} | outlier_rate={row['outlier_rate']:.2%}")

        print("\n--- topic_info ---")
        try:
            print(get_top_topics(run_dir, args.top_n).to_string(index=False))
        except Exception as e:
            print(f"Could not read topic_info: {e}")

        print("\n--- topic_words ---")
        try:
            print(get_top_words(run_dir, args.top_n).to_string(index=False))
        except Exception as e:
            print(f"Could not read topic_words: {e}")


if __name__ == "__main__":
    main()