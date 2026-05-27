#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd
from bertopic import BERTopic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to podcast_chunks_sw-de folder.")
    parser.add_argument("--strategy", default="c-tf-idf", choices=["c-tf-idf", "embeddings", "probabilities", "distributions"])
    parser.add_argument("--out-name", default="doc_topics_reassigned.parquet")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model_dir = run_dir / "bertopic_model"
    doc_path = run_dir / "doc_topics.parquet"

    docs_df = pd.read_parquet(doc_path)
    texts = docs_df["chunk_text"].astype(str).tolist()
    topics = docs_df["topic"].astype(int).tolist()

    print("Loading model:", model_dir)
    model = BERTopic.load(str(model_dir))

    before = (docs_df["topic"] == -1).mean()
    print(f"Outlier rate before: {before:.2%}")

    print("Reducing outliers with strategy:", args.strategy)
    new_topics = model.reduce_outliers(
        texts,
        topics,
        strategy=args.strategy,
    )

    out = docs_df.copy()
    out["topic_original"] = out["topic"]
    out["topic"] = new_topics
    out["was_reassigned_from_outlier"] = (out["topic_original"] == -1) & (out["topic"] != -1)

    after = (out["topic"] == -1).mean()
    reassigned = out["was_reassigned_from_outlier"].sum()

    print(f"Outlier rate after:  {after:.2%}")
    print(f"Reassigned chunks:   {reassigned}")

    out_path = run_dir / args.out_name
    out.to_parquet(out_path, index=False)
    print("Saved:", out_path)

    counts = out["topic"].value_counts().head(30)
    print("\nTop reassigned topic counts:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
