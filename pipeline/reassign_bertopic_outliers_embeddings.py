#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to podcast_chunks_sw-de folder.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Same embedding model used for the BERTopic run.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-name", default="doc_topics_reassigned_embeddings.parquet")
    parser.add_argument("--embeddings-cache", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model_dir = run_dir / "bertopic_model"
    doc_path = run_dir / "doc_topics.parquet"

    docs_df = pd.read_parquet(doc_path)
    texts = docs_df["chunk_text"].astype(str).tolist()
    topics = docs_df["topic"].astype(int).tolist()

    before = (docs_df["topic"] == -1).mean()
    print(f"Documents: {len(docs_df)}")
    print(f"Outlier rate before: {before:.2%}")

    print("Loading SentenceTransformer:", args.embedding_model)
    st_model = SentenceTransformer(args.embedding_model, device=args.device)

    print("Loading BERTopic model:", model_dir)
    model = BERTopic.load(str(model_dir), embedding_model=st_model)

    if args.embeddings_cache:
        cache_path = Path(args.embeddings_cache)
    else:
        cache_path = run_dir / "embeddings_for_outlier_reassignment.npy"

    if cache_path.exists():
        print("Loading cached embeddings:", cache_path)
        embeddings = np.load(cache_path)
    else:
        print("Encoding documents for outlier reassignment...")
        embeddings = st_model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        print("Saving embeddings cache:", cache_path)
        np.save(cache_path, embeddings)

    print("Reducing outliers with strategy: embeddings")
    new_topics = model.reduce_outliers(
        texts,
        topics,
        strategy="embeddings",
        embeddings=embeddings,
    )

    out = docs_df.copy()
    out["topic_original"] = out["topic"]
    out["topic"] = new_topics
    out["was_reassigned_from_outlier"] = (
        (out["topic_original"] == -1) & (out["topic"] != -1)
    )

    after = (out["topic"] == -1).mean()
    reassigned = int(out["was_reassigned_from_outlier"].sum())

    print(f"Outlier rate after:  {after:.2%}")
    print(f"Reassigned chunks:   {reassigned}")

    out_path = run_dir / args.out_name
    out.to_parquet(out_path, index=False)
    print("Saved:", out_path)

    print("\nTop topic counts after reassignment:")
    print(out["topic"].value_counts().head(30).to_string())


if __name__ == "__main__":
    main()
