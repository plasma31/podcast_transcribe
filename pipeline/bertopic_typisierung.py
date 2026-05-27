import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from stopwordsiso import stopwords as stopwords_iso


DEFAULT_TEXT_COLUMNS = ["ziele_combined_optimized"]
DELIMITER = "||"

DEFAULT_NAMES_DATASET_COUNTRIES = "DE,AT,CH,TR,PL,RO,RU,UA,FR,IT,ES,PT,NL,BE,GB,US"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERTopic typisierung for selected columns.")
    parser.add_argument(
        "--input",
        default="E:/LVR/output/bogenGlf_goals_joined_filtered.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="E:/LVR/bertopic_optimized",
        help="Output directory.",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional encoding.",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Optional delimiter (e.g., ',' or ';').",
    )

    parser.add_argument(
        "--text-columns",
        type=str,
        default=",".join(DEFAULT_TEXT_COLUMNS),
        help="Comma-separated list of text columns to model.",
    )

    parser.add_argument(
        "--stopwords",
        choices=["none", "de"],
        default="de",
        help="Stopwords mode for vectorizer.",
    )

    # UMAP parameters
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-n-components", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--umap-metric", type=str, default="cosine")
    parser.add_argument("--umap-random-state", type=int, default=42)

    # HDBSCAN parameters
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=15)
    parser.add_argument("--hdbscan-min-samples", type=int, default=1)
    parser.add_argument("--hdbscan-metric", type=str, default="euclidean")
    parser.add_argument("--hdbscan-cluster-selection-method", type=str, default="eom")

    # Vectorizer parameters
    parser.add_argument("--vectorizer-ngram-min", type=int, default=1)
    parser.add_argument("--vectorizer-ngram-max", type=int, default=2)
    parser.add_argument("--vectorizer-min-df", type=int, default=2)
    parser.add_argument(
        "--vectorizer-max-df",
        type=float,
        default=1.0,
        help="Max document frequency (float in (0,1] or int).",
    )

    parser.add_argument(
        "--nr-topics",
        type=int,
        default=None,
        help="Optional: reduce/merge topics after training to this number (e.g. 149).",
    )

    # Additional stopwords (comma-separated)
    parser.add_argument(
        "--extra-stopwords",
        type=str,
        default=None,
        help="Comma-separated extra stopwords to add (applies to stopword run).",
    )
    parser.add_argument(
        "--name-stopwords-file",
        type=str,
        default=None,
        help="Optional file path with additional name stopwords (one term per line).",
    )

    # Larger name lists via library (recommended over ad-hoc Kaggle downloads for reproducibility)
    parser.add_argument(
        "--use-names-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the 'names-dataset' library to add top-N first/last names per country.",
    )
    parser.add_argument(
        "--names-dataset-mode",
        choices=["top", "all"],
        default="top",
        help=(
            "How to build name stopwords from names-dataset: "
            "'top' uses get_top_names(top_n, ...) per country (fast, smaller); "
            "'all' scans the entire dataset and collects all names matching the selected countries (can be huge)."
        ),
    )
    parser.add_argument(
        "--names-dataset-countries",
        type=str,
        default=DEFAULT_NAMES_DATASET_COUNTRIES,
        help=(
            "Comma-separated ISO-3166 alpha2 country codes used by names-dataset (e.g. 'DE,AT,CH,TR'). "
            "Use 'ALL' to include names from all countries (only meaningful with --names-dataset-mode all)."
        ),
    )
    parser.add_argument(
        "--names-dataset-top-n",
        type=int,
        default=20000,
        help="Top-N names per gender (first names) and per country (last names) taken from names-dataset.",
    )
    parser.add_argument(
        "--names-dataset-min-country-prob",
        type=float,
        default=0.0,
        help=(
            "Only for --names-dataset-mode all: minimum country probability for a name to be included. "
            "0.0 includes any name that appears for the country at all; higher values reduce the list size."
        ),
    )
    return parser.parse_args()


def _parse_extra_stopwords(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [s.strip().lower() for s in value.split(",") if s.strip()]


def _load_stopwords_from_file(path: Optional[str]) -> List[str]:
    if not path:
        return []

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Name stopwords file not found: {p}")

    values: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        values.append(text.lower())
    return values


def _parse_country_list(value: str) -> Optional[List[str]]:
    raw = str(value or "").strip()
    if not raw:
        return []
    if raw.upper() in {"ALL", "*"}:
        return None

    items = [c.strip().upper() for c in raw.split(",") if c.strip()]
    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for c in items:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def get_name_stopwords_from_names_dataset_top(*, countries: List[str], top_n: int) -> List[str]:
    """Return a reasonably-sized stopword list of common first/last names.

    Uses names-dataset get_top_names() to avoid loading millions of names.
    """
    try:
        from names_dataset import NameDataset  # type: ignore
    except Exception:
        return []

    nd = NameDataset()
    stop: set[str] = set()

    def _add_first_names(data: object) -> None:
        if not isinstance(data, dict):
            return
        for _country, payload in data.items():
            if isinstance(payload, dict):
                for _gender, names in payload.items():
                    if isinstance(names, list):
                        for n in names:
                            if n:
                                stop.add(str(n).strip().lower())
            elif isinstance(payload, list):
                for n in payload:
                    if n:
                        stop.add(str(n).strip().lower())

    def _add_last_names(data: object) -> None:
        if not isinstance(data, dict):
            return
        for _country, payload in data.items():
            if isinstance(payload, list):
                for n in payload:
                    if n:
                        stop.add(str(n).strip().lower())

    for country in countries:
        # First names: dict like {'DE': {'M': [...], 'F': [...]}}
        _add_first_names(nd.get_top_names(n=top_n, use_first_names=True, country_alpha2=country))
        # Last names: dict like {'DE': ['Müller', 'Schmidt', ...]}
        _add_last_names(nd.get_top_names(n=top_n, use_first_names=False, country_alpha2=country))

    return sorted(s for s in stop if s)


def get_name_stopwords_from_names_dataset_all(
    *,
    countries: Optional[List[str]],
    min_country_prob: float,
) -> List[str]:
    """Return an exhaustive stopword list of names from names-dataset.

    IMPORTANT: This can be extremely large (hundreds of thousands to millions of stopwords),
    which will slow down CountVectorizer and can consume a lot of memory.

    If countries is None, includes names from all countries.
    If countries is a list, includes names where the name has a country probability >= min_country_prob
    for at least one of the selected countries.
    """
    try:
        from names_dataset import NameDataset  # type: ignore
    except Exception:
        return []

    nd = NameDataset()
    stop: set[str] = set()

    def _match_country(meta: object) -> bool:
        if countries is None:
            return True
        if not isinstance(meta, dict):
            return False
        country_map = meta.get("country")
        if not isinstance(country_map, dict):
            return False
        for c in countries:
            try:
                p = float(country_map.get(c, 0.0))
            except Exception:
                p = 0.0
            if p >= float(min_country_prob):
                return True
        return False

    for name, meta in getattr(nd, "first_names", {}).items():
        if name and _match_country(meta):
            stop.add(str(name).strip().lower())
    for name, meta in getattr(nd, "last_names", {}).items():
        if name and _match_country(meta):
            stop.add(str(name).strip().lower())

    return sorted(s for s in stop if s)


def build_stopwords(extra_stopwords: Sequence[str]) -> List[str]:
    base = set(stopwords_iso("de"))

    # Requested additions
    base.update({"herr", "frau", "herrn", "fraun", "frauen", "herren", "hr", "fr", "lb", "le"})

    # Common backslash/control-sequence artifacts
    base.update({"\\r", "\\n", "\\t", "\\", "\r", "\n", "\t"})

    for w in extra_stopwords:
        if w:
            base.add(str(w).strip().lower())

    return sorted(base)


def build_model(
    *,
    stopwords: Optional[List[str]],
    embedding_model: SentenceTransformer,
    umap_n_neighbors: int,
    umap_n_components: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    hdbscan_metric: str,
    hdbscan_cluster_selection_method: str,
    vectorizer_ngram_min: int,
    vectorizer_ngram_max: int,
    vectorizer_min_df: int,
    vectorizer_max_df: float,
) -> BERTopic:
    # Optimized configuration inspired by BERTopic best practices
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_random_state,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=hdbscan_cluster_selection_method,
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(vectorizer_ngram_min, vectorizer_ngram_max),
        stop_words=stopwords,
        min_df=vectorizer_min_df,
        max_df=vectorizer_max_df,
    )

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        calculate_probabilities=True,
        verbose=True,
    )


def preprocess_text(value: str) -> str:
    text = str(value)
    # Replace literal backslash sequences and actual control characters
    text = text.replace("\\r", " ").replace("\\n", " ").replace("\\t", " ")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def split_documents(series: pd.Series) -> List[str]:
    parts: List[str] = []
    for value in series.fillna("").astype(str).tolist():
        value = preprocess_text(value)
        if DELIMITER in value:
            parts.extend([v.strip() for v in value.split(DELIMITER) if v.strip()])
        else:
            if value.strip():
                parts.append(value.strip())
    return parts


def run_topic_model(
    *,
    df: pd.DataFrame,
    text_col: str,
    stopwords: Optional[List[str]],
    output_dir: Path,
    tag: str,
    embedding_model: SentenceTransformer,
    umap_n_neighbors: int,
    umap_n_components: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    hdbscan_metric: str,
    hdbscan_cluster_selection_method: str,
    vectorizer_ngram_min: int,
    vectorizer_ngram_max: int,
    vectorizer_min_df: int,
    vectorizer_max_df: float,
    nr_topics: Optional[int],
) -> None:
    texts = split_documents(df[text_col])
    model = build_model(
        stopwords=stopwords,
        embedding_model=embedding_model,
        umap_n_neighbors=umap_n_neighbors,
        umap_n_components=umap_n_components,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        umap_random_state=umap_random_state,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_metric=hdbscan_metric,
        hdbscan_cluster_selection_method=hdbscan_cluster_selection_method,
        vectorizer_ngram_min=vectorizer_ngram_min,
        vectorizer_ngram_max=vectorizer_ngram_max,
        vectorizer_min_df=vectorizer_min_df,
        vectorizer_max_df=vectorizer_max_df,
    )
    topics, probs = model.fit_transform(texts)

    if nr_topics is not None:
        model.reduce_topics(texts, nr_topics=nr_topics)
        topics, probs = model.transform(texts)

    out_base = output_dir / f"{text_col}_{tag}"
    out_base.mkdir(parents=True, exist_ok=True)

    # Save topic info
    topic_info = model.get_topic_info()
    topic_info.to_csv(out_base / "topic_info.csv", index=False)

    # Save document-level topics
    doc_topics = pd.DataFrame(
        {
            "doc_id": range(len(texts)),
            "topic": topics,
        }
    )
    doc_topics.to_csv(out_base / "doc_topics.csv", index=False)

    # Save probabilities (optional, can be large)
    if probs is not None:
        probs_df = pd.DataFrame(probs)
        probs_df.to_csv(out_base / "doc_topic_probs.csv", index=False)

    # Save top words per topic
    topics_words = []
    for topic_id in topic_info["Topic"].tolist():
        words = model.get_topic(topic_id)
        if words:
            topics_words.append(
                {
                    "topic": topic_id,
                    "words": ", ".join([w for w, _ in words]),
                }
            )
    pd.DataFrame(topics_words).to_csv(out_base / "topic_words.csv", index=False)

    # Save HTML visualizations
    try:
        fig_topics = model.visualize_topics()
        fig_topics.write_html(out_base / "topics_overview.html", include_plotlyjs="cdn")
    except Exception:
        pass

    try:
        fig_barchart = model.visualize_barchart()
        fig_barchart.write_html(out_base / "topics_barchart.html", include_plotlyjs="cdn")
    except Exception:
        pass

    try:
        fig_hierarchy = model.visualize_hierarchy()
        fig_hierarchy.write_html(out_base / "topics_hierarchy.html", include_plotlyjs="cdn")
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    read_kwargs = {}
    if args.encoding:
        read_kwargs["encoding"] = args.encoding
    if args.delimiter:
        read_kwargs["sep"] = args.delimiter

    if args.delimiter:
        df = pd.read_csv(input_path, **read_kwargs)
    else:
        # Try auto-detect delimiter for robustness
        df = pd.read_csv(input_path, sep=None, engine="python", **read_kwargs)

    text_columns = [c.strip() for c in str(args.text_columns).split(",") if c.strip()]
    df = df[[c for c in text_columns if c in df.columns]].copy()

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    extra_stopwords = _parse_extra_stopwords(args.extra_stopwords)

    # Explicit requested additions
    extra_stopwords.extend(["müller", "özdal", "özsoy"])

    # Larger lists via names-dataset (top-N per country)
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
                raise ValueError(
                    "--names-dataset-countries ALL is only supported with --names-dataset-mode all"
                )
            extra_stopwords.extend(
                get_name_stopwords_from_names_dataset_top(
                    countries=countries,
                    top_n=int(args.names_dataset_top_n),
                )
            )
        after = len(extra_stopwords)
        print(
            f"names-dataset stopwords added: {after - before} "
            f"(mode={args.names_dataset_mode}, countries={'ALL' if countries is None else len(countries)})"
        )

    # Optional user-provided names file (one term per line)
    extra_stopwords.extend(_load_stopwords_from_file(args.name_stopwords_file))

    if args.stopwords == "de":
        stopwords = build_stopwords(extra_stopwords)
        tag = "sw-de"
    else:
        stopwords = None
        tag = "sw-none"

    for text_col in df.columns.tolist():
        run_topic_model(
            df=df,
            text_col=text_col,
            stopwords=stopwords,
            output_dir=output_dir,
            tag=tag,
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
            nr_topics=args.nr_topics,
        )


if __name__ == "__main__":
    main()
