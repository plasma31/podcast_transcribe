#!/usr/bin/env python3
"""
greedy_grid_search_bertopic.py

Greedy grid search over UMAP n_neighbors and HDBSCAN min_cluster_size.
Optimises coherence score (higher = better) and noise count (-1 topics, lower = better).

Token filtering:
  < token_min  → excluded
  > token_max  → split into chunks of token_max (tail chunks < token_min are also excluded)

Output root: E:\\greedy_grid_search\\<timestamp>_gridsearch\\
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords as stopwords_iso
from umap import UMAP

try:
    from bertopic.representation import KeyBERTInspired
    HAS_KEYBERT = True
except ImportError:
    HAS_KEYBERT = False

try:
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOKEN_MIN = 50
DEFAULT_TOKEN_MAX = 128  # paraphrase-multilingual-MiniLM-L12-v2 max_seq_length
DELIMITER = "||"
DEFAULT_NAMES_COUNTRIES = "DE,AT,CH,TR,PL,RO,RU,UA,FR,IT,ES,PT,NL,BE,GB,US"
DEFAULT_NN_GRID = "10,15,20,30,50"
DEFAULT_MCS_GRID = "10,20,30,50,80"


# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Greedy grid search BERTopic – coherence + noise optimisation"
    )
    p.add_argument("--input", default="E:/LVR/output/bogenGlf_goals_joined_filtered.csv")
    p.add_argument("--output-dir", default="E:/greedy_grid_search")
    p.add_argument("--encoding", default=None)
    p.add_argument("--delimiter", default=None)
    p.add_argument("--text-columns", default="ziele_combined_optimized",
                   help="Comma-separated column names to model.")

    # Token filtering
    p.add_argument("--token-min", type=int, default=DEFAULT_TOKEN_MIN,
                   help="Exclude docs with fewer tokens (default 50).")
    p.add_argument("--token-max", type=int, default=DEFAULT_TOKEN_MAX,
                   help="Split docs with more tokens (default 128 = model's max_seq_length).")

    # Grid search
    p.add_argument("--nn-grid", default=DEFAULT_NN_GRID,
                   help="Comma-separated UMAP n_neighbors values for grid search.")
    p.add_argument("--mcs-grid", default=DEFAULT_MCS_GRID,
                   help="Comma-separated HDBSCAN min_cluster_size values for grid search.")
    p.add_argument("--coherence-weight", type=float, default=0.5,
                   help="Weight of coherence score in composite score [0..1].")

    # Fixed UMAP
    p.add_argument("--umap-n-components", type=int, default=5)
    p.add_argument("--umap-min-dist", type=float, default=0.01)
    p.add_argument("--umap-metric", default="cosine")
    p.add_argument("--umap-random-state", type=int, default=42)

    # Fixed HDBSCAN
    p.add_argument("--hdbscan-min-samples", type=int, default=1)
    p.add_argument("--hdbscan-metric", default="euclidean")
    p.add_argument("--hdbscan-cluster-selection-method", default="eom")

    # Vectorizer
    p.add_argument("--vectorizer-ngram-min", type=int, default=1)
    p.add_argument("--vectorizer-ngram-max", type=int, default=2)
    p.add_argument("--vectorizer-min-df", type=int, default=2)
    p.add_argument("--vectorizer-max-df", type=float, default=1.0)
    p.add_argument("--nr-topics", type=int, default=None)

    # Stopwords
    p.add_argument("--stopwords", choices=["none", "de"], default="de")
    p.add_argument("--extra-stopwords", default=None)
    p.add_argument("--name-stopwords-file", default=None)
    p.add_argument("--use-names-dataset", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--names-dataset-mode", choices=["top", "all"], default="top")
    p.add_argument("--names-dataset-countries", default=DEFAULT_NAMES_COUNTRIES)
    p.add_argument("--names-dataset-top-n", type=int, default=20000)
    p.add_argument("--names-dataset-min-country-prob", type=float, default=0.0)

    # Output options
    p.add_argument("--save-noise-docs", action="store_true", default=False,
                   help="Save -1 topic documents to CSV per run (default: off).")
    p.add_argument("--keep-representative-docs",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prob-threshold", type=float, default=0.1)

    # KeyBERT finetuning
    p.add_argument("--use-keybert", action="store_true", default=False,
                   help="Use KeyBERTInspired representation model.")

    return p.parse_args()


# ── Stopwords ──────────────────────────────────────────────────────────────────
def _parse_extra_stopwords(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [s.strip().lower() for s in value.split(",") if s.strip()]


def _load_stopwords_from_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stopwords file not found: {p}")
    return [
        line.strip().lower()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _parse_country_list(value: str) -> Optional[List[str]]:
    raw = str(value or "").strip()
    if not raw:
        return []
    if raw.upper() in {"ALL", "*"}:
        return None
    seen: set = set()
    out: List[str] = []
    for c in raw.split(","):
        c = c.strip().upper()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def get_name_stopwords_top(*, countries: List[str], top_n: int) -> List[str]:
    try:
        from names_dataset import NameDataset  # type: ignore
    except Exception:
        return []
    nd = NameDataset()
    stop: set = set()

    def _add(data: object) -> None:
        if not isinstance(data, dict):
            return
        for v in data.values():
            if isinstance(v, dict):
                for names in v.values():
                    if isinstance(names, list):
                        stop.update(str(n).strip().lower() for n in names if n)
            elif isinstance(v, list):
                stop.update(str(n).strip().lower() for n in v if n)

    for country in countries:
        _add(nd.get_top_names(n=top_n, use_first_names=True, country_alpha2=country))
        _add(nd.get_top_names(n=top_n, use_first_names=False, country_alpha2=country))
    return sorted(s for s in stop if s)


def build_stopwords(extra: Sequence[str]) -> List[str]:
    base = set(stopwords_iso("de"))
    base.update({"herr", "frau", "herrn", "fraun", "frauen", "herren", "hr", "fr",
                 "lb", "le", "\\r", "\\n", "\\t"})
    base.update(str(w).strip().lower() for w in extra if w)
    return sorted(base)


# ── Text preprocessing & token filtering ──────────────────────────────────────
def preprocess_text(value: str) -> str:
    text = str(value)
    for seq in ("\\r", "\\n", "\\t", "\r", "\n", "\t"):
        text = text.replace(seq, " ")
    return " ".join(text.split())


def filter_and_split_by_tokens(
    series: pd.Series,
    tokenizer,
    token_min: int,
    token_max: int,
) -> Tuple[List[str], int, int]:
    """Return (texts, n_excluded, n_split_docs)."""
    n_excluded = 0
    n_split = 0
    result: List[str] = []

    for raw in series.fillna("").astype(str).tolist():
        text = preprocess_text(raw)
        if not text:
            continue
        candidates = (
            [v.strip() for v in text.split(DELIMITER) if v.strip()]
            if DELIMITER in text else [text]
        )
        for candidate in candidates:
            if tokenizer is not None:
                encoded = tokenizer(candidate, truncation=False, padding=False)
                ids = encoded.get("input_ids", [])
                n_tokens = len(ids)
            else:
                ids = candidate.split()
                n_tokens = len(ids)

            if n_tokens < token_min:
                n_excluded += 1
            elif n_tokens > token_max:
                n_split += 1
                for start in range(0, n_tokens, token_max):
                    chunk_ids = ids[start: start + token_max]
                    if len(chunk_ids) < token_min:
                        n_excluded += 1
                        continue
                    if tokenizer is not None:
                        chunk_text = preprocess_text(
                            tokenizer.decode(
                                chunk_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                            )
                        )
                    else:
                        chunk_text = " ".join(chunk_ids)
                    if chunk_text:
                        result.append(chunk_text)
            else:
                result.append(candidate)

    return result, n_excluded, n_split


# ── Coherence (gensim C_v) ─────────────────────────────────────────────────────
def compute_coherence(model: BERTopic, texts: List[str]) -> float:
    if not HAS_GENSIM:
        print("  [warn] gensim not installed – coherence score will be NaN.")
        return float("nan")

    tokenized = [t.lower().split() for t in texts]
    dictionary = Dictionary(tokenized)
    vocab = set(dictionary.token2id.keys())

    # Topic-Wörter robust aufbereiten:
    #  - leere/whitespace-Wörter entfernen (KeyBERTInspired liefert teils '')
    #  - nur Wörter behalten, die im Dictionary vorkommen (Bigramme/OOV droppen,
    #    sonst rechnet gensim mit Lücken oder wirft Fehler)
    #  - Topics mit < 2 gültigen Wörtern überspringen (sonst "unable to interpret topic")
    topic_words: List[List[str]] = []
    skipped = 0
    for _, row in model.get_topic_info().iterrows():
        if int(row["Topic"]) == -1:
            continue
        words = model.get_topic(int(row["Topic"])) or []
        clean = [
            str(w).strip().lower()
            for w, _ in words[:10]
            if w and str(w).strip()
        ]
        clean = [w for w in clean if w in vocab]
        if len(clean) >= 2:
            topic_words.append(clean)
        else:
            skipped += 1

    if len(topic_words) < 2:
        print(f"  [warn] Coherence: zu wenige gültige Topics ({len(topic_words)}, {skipped} übersprungen).")
        return float("nan")

    try:
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        score = float(cm.get_coherence())
        if skipped:
            print(f"  [info] Coherence über {len(topic_words)} Topics ({skipped} ohne gültige Wörter übersprungen).")
        return score
    except Exception as exc:
        print(f"  [warn] Coherence calculation failed: {exc}")
        return float("nan")


# ── BERTopic builder ───────────────────────────────────────────────────────────
def build_bertopic(
    *,
    embedding_model: SentenceTransformer,
    stopwords: Optional[List[str]],
    n_neighbors: int,
    n_components: int,
    min_dist: float,
    umap_metric: str,
    random_state: int,
    min_cluster_size: int,
    min_samples: int,
    hdbscan_metric: str,
    cluster_selection_method: str,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
    use_keybert: bool = False,
) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=umap_metric,
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        ngram_range=(ngram_min, ngram_max),
        stop_words=stopwords,
        min_df=min_df,
        max_df=max_df,
    )
    kwargs: Dict[str, Any] = dict(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=10,
        calculate_probabilities=True,
        verbose=False,
    )
    if use_keybert and HAS_KEYBERT:
        kwargs["representation_model"] = KeyBERTInspired()
    return BERTopic(**kwargs)


# ── Single run ─────────────────────────────────────────────────────────────────
def run_single(
    *,
    texts: List[str],
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    stopwords: Optional[List[str]],
    n_neighbors: int,
    min_cluster_size: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    eff_nn = min(n_neighbors, max(len(texts) - 1, 1))
    eff_mcs = min(min_cluster_size, len(texts))

    model = build_bertopic(
        embedding_model=embedding_model,
        stopwords=stopwords,
        n_neighbors=eff_nn,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        random_state=args.umap_random_state,
        min_cluster_size=eff_mcs,
        min_samples=args.hdbscan_min_samples,
        hdbscan_metric=args.hdbscan_metric,
        cluster_selection_method=args.hdbscan_cluster_selection_method,
        ngram_min=args.vectorizer_ngram_min,
        ngram_max=args.vectorizer_ngram_max,
        min_df=args.vectorizer_min_df,
        max_df=args.vectorizer_max_df,
        use_keybert=args.use_keybert,
    )

    topics, probs = model.fit_transform(texts, embeddings=embeddings)

    if args.nr_topics is not None:
        model.reduce_topics(texts, nr_topics=args.nr_topics)
        topics = list(model.topics_)
        probs = model.probabilities_

    topics_arr = np.array(topics)
    n_noise = int((topics_arr == -1).sum())
    noise_frac = n_noise / max(len(topics), 1)
    n_topics = int((model.get_topic_info()["Topic"] != -1).sum())
    coherence = compute_coherence(model, texts)

    return {
        "n_neighbors": n_neighbors,
        "min_cluster_size": min_cluster_size,
        "n_topics": n_topics,
        "n_noise": n_noise,
        "noise_frac": round(noise_frac, 4),
        "coherence": coherence,
        "model": model,
        "topics": list(topics),
        "probs": probs,
    }


# ── Composite score ────────────────────────────────────────────────────────────
def composite_scores(results: List[Dict], alpha: float) -> List[float]:
    """Normalised: alpha * coherence_norm + (1-alpha) * (1 - noise_norm). Higher = better."""
    coherences = [r["coherence"] for r in results]
    noise_fracs = [r["noise_frac"] for r in results]

    valid_c = [c for c in coherences if not np.isnan(c)]
    c_min, c_max = (min(valid_c), max(valid_c)) if valid_c else (0.0, 1.0)
    c_range = max(c_max - c_min, 1e-9)
    c_norm = [(0.0 if np.isnan(c) else (c - c_min) / c_range) for c in coherences]

    n_min, n_max = min(noise_fracs), max(noise_fracs)
    n_range = max(n_max - n_min, 1e-9)
    n_norm = [1.0 - (n - n_min) / n_range for n in noise_fracs]

    return [alpha * c + (1.0 - alpha) * n for c, n in zip(c_norm, n_norm)]


# ── Greedy grid search ─────────────────────────────────────────────────────────
def greedy_grid_search(
    *,
    texts: List[str],
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    stopwords: Optional[List[str]],
    nn_grid: List[int],
    mcs_grid: List[int],
    args: argparse.Namespace,
) -> Tuple[List[Dict], int, int]:
    alpha = args.coherence_weight
    default_mcs = mcs_grid[len(mcs_grid) // 2]

    print(f"\n── Phase 1: optimise n_neighbors  (fixed min_cluster_size={default_mcs}) ──")
    phase1: List[Dict] = []
    for nn in nn_grid:
        print(f"  nn={nn:<4}  mcs={default_mcs:<4} ... ", end="", flush=True)
        res = run_single(
            texts=texts, embeddings=embeddings, embedding_model=embedding_model,
            stopwords=stopwords, n_neighbors=nn, min_cluster_size=default_mcs, args=args,
        )
        phase1.append(res)
        print(f"topics={res['n_topics']}  noise={res['n_noise']}  coherence={res['coherence']:.4f}")

    scores1 = composite_scores(phase1, alpha)
    best_nn = phase1[int(np.argmax(scores1))]["n_neighbors"]
    print(f"  → best n_neighbors = {best_nn}  (composite={max(scores1):.4f})")

    print(f"\n── Phase 2: optimise min_cluster_size  (fixed n_neighbors={best_nn}) ──")
    phase2: List[Dict] = []
    for mcs in mcs_grid:
        existing = next(
            (r for r in phase1 if r["n_neighbors"] == best_nn and r["min_cluster_size"] == mcs),
            None,
        )
        if existing:
            phase2.append(existing)
            print(f"  nn={best_nn:<4}  mcs={mcs:<4}  (reused) topics={existing['n_topics']}  noise={existing['n_noise']}  coherence={existing['coherence']:.4f}")
            continue
        print(f"  nn={best_nn:<4}  mcs={mcs:<4} ... ", end="", flush=True)
        res = run_single(
            texts=texts, embeddings=embeddings, embedding_model=embedding_model,
            stopwords=stopwords, n_neighbors=best_nn, min_cluster_size=mcs, args=args,
        )
        phase2.append(res)
        print(f"topics={res['n_topics']}  noise={res['n_noise']}  coherence={res['coherence']:.4f}")

    scores2 = composite_scores(phase2, alpha)
    best_mcs = phase2[int(np.argmax(scores2))]["min_cluster_size"]
    print(f"  → best min_cluster_size = {best_mcs}  (composite={max(scores2):.4f})")

    seen: set = set()
    all_results: List[Dict] = []
    for r in phase1 + phase2:
        key = (r["n_neighbors"], r["min_cluster_size"])
        if key not in seen:
            seen.add(key)
            all_results.append(r)

    # Composite-Score über ALLE Konfigurationen neu normalisieren und anhängen,
    # damit der Kompromiss Kohärenz <-> Noise pro Konfiguration sichtbar/vergleichbar ist.
    final_scores = composite_scores(all_results, alpha)
    for r, s in zip(all_results, final_scores):
        r["composite_score"] = round(float(s), 4)

    # Bester Gesamt-Kompromiss (über alle getesteten Konfigurationen)
    best_overall = all_results[int(np.argmax(final_scores))]
    best_nn = best_overall["n_neighbors"]
    best_mcs = best_overall["min_cluster_size"]
    print(
        f"\n  → Bester Kompromiss (alpha={alpha}): nn={best_nn}, mcs={best_mcs} | "
        f"coherence={best_overall['coherence']:.4f}  noise_frac={best_overall['noise_frac']:.3f}  "
        f"composite={best_overall['composite_score']:.4f}"
    )

    return all_results, best_nn, best_mcs


# ── Visualisations ─────────────────────────────────────────────────────────────
def _param_tag(nn: int, mcs: int) -> str:
    return f"nn{nn}_mcs{mcs}"


def plot_grid_search_overview(results: List[Dict], out_path: Path) -> None:
    rows = [
        {
            "label": _param_tag(r["n_neighbors"], r["min_cluster_size"]),
            "n_neighbors": r["n_neighbors"],
            "min_cluster_size": r["min_cluster_size"],
            "coherence": r["coherence"],
            "noise_pct": round(r["noise_frac"] * 100.0, 1),
            "composite_score": r.get("composite_score", float("nan")),
            "n_topics": r["n_topics"],
        }
        for r in sorted(results, key=lambda x: (x["n_neighbors"], x["min_cluster_size"]))
    ]
    df = pd.DataFrame(rows)
    best_label = df.loc[df["composite_score"].idxmax(), "label"] if df["composite_score"].notna().any() else None

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Coherence Score (höher = besser)",
            "Noise-Anteil -1 in % (niedriger = besser)",
            "Composite Score – Kompromiss (höher = besser)",
        ],
    )
    fig.add_trace(
        go.Bar(x=df["label"], y=df["coherence"],
               marker_color="steelblue", name="Coherence",
               text=df["coherence"].round(3), textposition="outside"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=df["label"], y=df["noise_pct"],
               marker_color="tomato", name="Noise %",
               text=df["noise_pct"].astype(str) + " %", textposition="outside"),
        row=1, col=2,
    )
    comp_colors = ["seagreen" if lbl == best_label else "lightgray" for lbl in df["label"]]
    fig.add_trace(
        go.Bar(x=df["label"], y=df["composite_score"],
               marker_color=comp_colors, name="Composite",
               text=df["composite_score"].round(3), textposition="outside"),
        row=1, col=3,
    )
    title = "Greedy Grid Search – Kompromiss Kohärenz ↔ Noise"
    if best_label:
        title += f"  |  bester Kompromiss: {best_label}"
    fig.update_layout(title=title, height=540, showlegend=False)
    fig.update_xaxes(tickangle=45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  Saved: {out_path.name}")


def plot_top10_topics(model: BERTopic, out_path: Path, tag: str = "") -> None:
    """Top-10 Topics als RELATIVE Häufigkeit (% aller Dokumente, inkl. -1 im Nenner)."""
    ti = model.get_topic_info()
    total_docs = int(ti["Count"].sum())  # alle Dokumente inkl. Noise (-1)
    top10 = ti[ti["Topic"] != -1].head(10).sort_values("Count", ascending=True).copy()
    top10["rel_freq"] = top10["Count"] / max(total_docs, 1) * 100.0
    fig = go.Figure(go.Bar(
        x=top10["rel_freq"], y=top10["Name"].astype(str),
        orientation="h", marker_color="steelblue",
        text=top10["rel_freq"].round(1).astype(str) + " %", textposition="outside",
    ))
    fig.update_layout(
        title=f"Top 10 Topics – relative Häufigkeit (% aller Docs)  {tag}",
        xaxis_title="Anteil an allen Dokumenten [%]", yaxis_title="Topic", height=500,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_topic_distribution(model: BERTopic, out_path: Path, tag: str = "") -> pd.DataFrame:
    """Zusätzliche Auswertung: relative Häufigkeit ALLER Topics (inkl. Noise -1).

    Gibt das DataFrame zurück, damit es zusätzlich als CSV gespeichert werden kann.
    """
    ti = model.get_topic_info().copy()
    total_docs = int(ti["Count"].sum())
    ti["rel_freq_pct"] = (ti["Count"] / max(total_docs, 1) * 100.0).round(2)
    ti["is_noise"] = ti["Topic"] == -1

    plot_df = ti.sort_values("Count", ascending=False).copy()
    colors = ["tomato" if noise else "steelblue" for noise in plot_df["is_noise"]]
    fig = go.Figure(go.Bar(
        x=plot_df["Name"].astype(str), y=plot_df["rel_freq_pct"],
        marker_color=colors,
        text=plot_df["rel_freq_pct"].astype(str) + " %", textposition="outside",
    ))
    fig.update_layout(
        title=f"Topic-Verteilung – relative Häufigkeit aller Topics (rot = Noise -1)  {tag}",
        xaxis_title="Topic", yaxis_title="Anteil an allen Dokumenten [%]", height=600,
    )
    fig.update_xaxes(tickangle=45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return ti[["Topic", "Name", "Count", "rel_freq_pct", "is_noise"]]


def plot_topic_correlation(model: BERTopic, out_path: Path, tag: str = "") -> None:
    try:
        import scipy.sparse as sp
        from sklearn.metrics.pairwise import cosine_similarity

        ctfidf = model.c_tf_idf_
        if sp.issparse(ctfidf):
            ctfidf = ctfidf.toarray()

        ti = model.get_topic_info()

        def _compact_label(topic_id: int) -> str:
            """Kompakt: 'ID_wort1_wort2' (nur die ersten 2 Topic-Wörter)."""
            words = model.get_topic(topic_id) or []
            top2 = [w for w, _ in words[:2]]
            return f"{topic_id}_" + "_".join(top2) if top2 else str(topic_id)

        labels = [_compact_label(int(r["Topic"])) for _, r in ti.iterrows() if int(r["Topic"]) != -1]
        # Row 0 of c_tf_idf_ corresponds to topic -1 in BERTopic
        matrix = ctfidf[1:, :] if ctfidf.shape[0] > 1 else ctfidf
        labels = labels[: matrix.shape[0]]
        sim = cosine_similarity(matrix)

        # Limit to first 50 topics for readability
        n = min(len(labels), 50)
        sim, labels = sim[:n, :n], labels[:n]

        fig = go.Figure(go.Heatmap(
            z=sim, x=labels, y=labels,
            colorscale="RdBu", zmin=-1, zmax=1,
        ))
        fig.update_layout(
            title=f"Topic Similarity – Cosine of c-TF-IDF  {tag}",
            height=700, xaxis=dict(tickangle=45),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), include_plotlyjs="cdn")
    except Exception as exc:
        print(f"  [warn] Topic correlation plot failed: {exc}")


def plot_clusters(
    model: BERTopic,
    texts: List[str],
    topics: List[int],
    embeddings: np.ndarray,
    out_path: Path,
    tag: str = "",
) -> None:
    try:
        reducer = UMAP(n_components=2, random_state=42, metric="cosine", verbose=False)
        coords = reducer.fit_transform(embeddings)

        sample_size = min(len(texts), 5000)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), size=sample_size, replace=False) if len(texts) > sample_size else np.arange(len(texts))

        df = pd.DataFrame({
            "x": coords[idx, 0],
            "y": coords[idx, 1],
            "topic": [str(topics[i]) for i in idx],
            "text": [texts[i][:80] + ("…" if len(texts[i]) > 80 else "") for i in idx],
        })
        fig = px.scatter(
            df, x="x", y="y", color="topic",
            hover_data={"text": True, "x": False, "y": False},
            title=f"Document Clusters (UMAP 2-D)  {tag}",
        )
        fig.update_traces(marker_size=3)
        fig.update_layout(height=700)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), include_plotlyjs="cdn")
    except Exception as exc:
        print(f"  [warn] Cluster plot failed: {exc}")


# ── Save a single run's outputs ────────────────────────────────────────────────
def save_run_outputs(
    *,
    run_dir: Path,
    model: BERTopic,
    topics: List[int],
    probs: Any,
    texts: List[str],
    embeddings: np.ndarray,
    n_neighbors: int,
    min_cluster_size: int,
    coherence: float,
    n_noise: int,
    n_topics: int,
    args: argparse.Namespace,
) -> None:
    tag = _param_tag(n_neighbors, min_cluster_size)
    run_dir.mkdir(parents=True, exist_ok=True)

    ti = model.get_topic_info()
    if not args.keep_representative_docs and "Representative_Docs" in ti.columns:
        ti = ti.drop(columns=["Representative_Docs"])
    ti.to_csv(run_dir / f"topic_info_{tag}.csv", index=False)

    pd.DataFrame({"doc_id": range(len(topics)), "topic": topics}).to_csv(
        run_dir / f"doc_topics_{tag}.csv", index=False
    )

    words_rows = []
    for _, row in model.get_topic_info().iterrows():
        tid = int(row["Topic"])
        words = model.get_topic(tid)
        if words:
            words_rows.append({"topic": tid, "words": ", ".join(w for w, _ in words)})
    pd.DataFrame(words_rows).to_csv(run_dir / f"topic_words_{tag}.csv", index=False)

    # Noise docs (optional)
    if args.save_noise_docs:
        noise_idx = [i for i, t in enumerate(topics) if t == -1]
        pd.DataFrame({"doc_id": noise_idx, "text": [texts[i] for i in noise_idx]}).to_csv(
            run_dir / f"noise_docs_{tag}.csv", index=False
        )

    # Probs
    if probs is not None:
        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 2:
            pd.DataFrame(probs_arr).to_csv(run_dir / f"doc_topic_probs_{tag}.csv", index=False)

    # Metrics
    pd.DataFrame([{
        "n_neighbors": n_neighbors,
        "min_cluster_size": min_cluster_size,
        "n_topics": n_topics,
        "n_noise": n_noise,
        "noise_frac": round(n_noise / max(len(topics), 1), 4),
        "coherence": coherence,
    }]).to_csv(run_dir / f"metrics_{tag}.csv", index=False)

    # BERTopic HTML visualisations
    for fname, fn in [
        (f"topics_overview_{tag}.html", model.visualize_topics),
        (f"topics_barchart_{tag}.html", model.visualize_barchart),
        (f"topics_hierarchy_{tag}.html", model.visualize_hierarchy),
    ]:
        try:
            fig = fn()
            fig.write_html(str(run_dir / fname), include_plotlyjs="cdn")
        except Exception:
            pass

    # Custom Plotly
    plot_top10_topics(model, run_dir / f"top10_topics_{tag}.html", tag=tag)
    plot_topic_correlation(model, run_dir / f"topic_correlation_{tag}.html", tag=tag)
    plot_clusters(model, texts, topics, embeddings, run_dir / f"clusters_{tag}.html", tag=tag)

    # Zusätzliche Auswertung: relative Topic-Verteilung (Plot + CSV)
    dist_df = plot_topic_distribution(model, run_dir / f"topic_distribution_{tag}.html", tag=tag)
    dist_df.to_csv(run_dir / f"topic_distribution_{tag}.csv", index=False)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    cmd_str = "python " + " ".join(sys.argv)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / f"{ts}_gridsearch"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save command
    (session_dir / "command.txt").write_text(cmd_str, encoding="utf-8")
    print(f"Session: {session_dir}")
    print(f"Command: {cmd_str}")

    # Load data
    read_kwargs: Dict[str, Any] = {}
    if args.encoding:
        read_kwargs["encoding"] = args.encoding
    if args.delimiter:
        read_kwargs["sep"] = args.delimiter
        df = pd.read_csv(args.input, **read_kwargs)
    else:
        df = pd.read_csv(args.input, sep=None, engine="python", **read_kwargs)

    text_columns = [c.strip() for c in str(args.text_columns).split(",") if c.strip()]
    df = df[[c for c in text_columns if c in df.columns]].copy()
    if df.empty or df.columns.empty:
        print("ERROR: No matching text columns found in input.")
        sys.exit(1)

    # Embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    tokenizer = getattr(embedding_model, "tokenizer", None)

    # Stopwords
    extra_sw = _parse_extra_stopwords(args.extra_stopwords)
    extra_sw.extend(["müller", "özdal", "özsoy"])

    if args.use_names_dataset:
        countries = _parse_country_list(args.names_dataset_countries)
        if args.names_dataset_mode == "top" and countries is not None:
            extra_sw.extend(get_name_stopwords_top(countries=countries, top_n=args.names_dataset_top_n))
        elif args.names_dataset_mode == "all":
            try:
                from names_dataset import NameDataset  # type: ignore
                nd = NameDataset()
                for name in list(getattr(nd, "first_names", {}).keys())[:200_000]:
                    extra_sw.append(str(name).strip().lower())
            except Exception:
                pass

    extra_sw.extend(_load_stopwords_from_file(args.name_stopwords_file))
    stopwords_list = build_stopwords(extra_sw) if args.stopwords == "de" else None

    # Grids
    nn_grid = sorted({int(x) for x in str(args.nn_grid).split(",") if x.strip()})
    mcs_grid = sorted({int(x) for x in str(args.mcs_grid).split(",") if x.strip()})

    # Save params
    (session_dir / "run_params.json").write_text(
        json.dumps({"args": vars(args), "nn_grid": nn_grid, "mcs_grid": mcs_grid, "ts": ts},
                   indent=2, default=str),
        encoding="utf-8",
    )

    for text_col in df.columns.tolist():
        print(f"\n{'='*60}")
        print(f"Column: {text_col}")

        texts, n_excl, n_split = filter_and_split_by_tokens(
            df[text_col], tokenizer=tokenizer,
            token_min=args.token_min, token_max=args.token_max,
        )
        print(
            f"Token filter: {n_excl} excluded (<{args.token_min} tok), "
            f"{n_split} split (>{args.token_max} tok), "
            f"{len(texts)} docs remaining"
        )

        if len(texts) < 10:
            print(f"  [skip] Only {len(texts)} documents after filtering – too few.")
            continue

        print("Computing embeddings (shared across all runs) ...")
        embeddings = embedding_model.encode(
            texts, show_progress_bar=True, batch_size=64,
            convert_to_numpy=True,
        )

        all_results, best_nn, best_mcs = greedy_grid_search(
            texts=texts, embeddings=embeddings, embedding_model=embedding_model,
            stopwords=stopwords_list, nn_grid=nn_grid, mcs_grid=mcs_grid, args=args,
        )

        # Grid search summary CSV + overview plot
        summary_rows = [
            {k: v for k, v in r.items() if k not in ("model", "topics", "probs")}
            for r in all_results
        ]
        pd.DataFrame(summary_rows).to_csv(session_dir / "grid_search_results.csv", index=False)
        plot_grid_search_overview(all_results, session_dir / "grid_search_overview.html")

        # Per-run outputs
        print("\nSaving per-run outputs ...")
        for res in all_results:
            tag = _param_tag(res["n_neighbors"], res["min_cluster_size"])
            run_dir = session_dir / "runs" / tag
            save_run_outputs(
                run_dir=run_dir, model=res["model"],
                topics=res["topics"], probs=res["probs"],
                texts=texts, embeddings=embeddings,
                n_neighbors=res["n_neighbors"], min_cluster_size=res["min_cluster_size"],
                coherence=res["coherence"], n_noise=res["n_noise"], n_topics=res["n_topics"],
                args=args,
            )
            print(f"  Saved run: {run_dir.name}")

        # Best run in dedicated folder
        best_tag = _param_tag(best_nn, best_mcs)
        best_res = next(
            r for r in all_results
            if r["n_neighbors"] == best_nn and r["min_cluster_size"] == best_mcs
        )
        best_dir = session_dir / f"best_{best_tag}"
        save_run_outputs(
            run_dir=best_dir, model=best_res["model"],
            topics=best_res["topics"], probs=best_res["probs"],
            texts=texts, embeddings=embeddings,
            n_neighbors=best_nn, min_cluster_size=best_mcs,
            coherence=best_res["coherence"], n_noise=best_res["n_noise"],
            n_topics=best_res["n_topics"], args=args,
        )

        print(f"\nBest config: nn={best_nn}, mcs={best_mcs}")
        print(f"  Topics: {best_res['n_topics']}  |  Noise: {best_res['n_noise']}  |  Coherence: {best_res['coherence']:.4f}")
        print(f"  Best run dir: {best_dir}")

    print(f"\nAll outputs: {session_dir}")


if __name__ == "__main__":
    main()
