#!/usr/bin/env python3
"""Compute corpus statistics and generate thesis figures.

Run with the full venv (has matplotlib + pandas):
    .venv/bin/python docs/thesis/_make_stats_and_figures.py

Outputs:
    docs/thesis/corpus_stats.json     - computed numbers used in the prose
    docs/thesis/figures/*.png         - static figures embedded in the chapter
This script is reproducible and reads only local pipeline outputs.
"""
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "savefig.bbox": "tight"})
BASELINE = ROOT / "outputs" / "bertopic" / "podcast_chunks_sw-de"

stats = {}

# ---------------------------------------------------------------- manifest
m = pd.read_parquet(ROOT / "outputs/state/manifest.parquet")
stats["manifest"] = {
    "episodes_total": int(len(m)),
    "podcasts": int(m["podcast_folder"].nunique()),
    "status": {k: int(v) for k, v in m["status"].value_counts().items()},
    "transcription_compute_hours": round(
        float(m.loc[m.status == "done", "runtime_sec"].astype(float).sum()) / 3600.0, 1
    ),
    "mean_episode_runtime_sec": round(
        float(m.loc[m.status == "done", "runtime_sec"].astype(float).mean()), 1
    ),
}

# ---------------------------------------------------------------- episodes
ep_files = glob.glob(str(ROOT / "outputs/parquet/episodes/*.parquet"))
ep = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
stats["episodes"] = {
    "n_episode_parquet": len(ep_files),
    "languages": {k: int(v) for k, v in ep["whisper_language"].value_counts().head(10).items()},
    "total_segments": int(ep["n_segments"].sum()),
    "mean_segments_per_episode": round(float(ep["n_segments"].mean()), 1),
    "mean_speakers_per_episode": round(float(ep["n_speakers"].mean()), 2),
    "median_speakers_per_episode": int(ep["n_speakers"].median()),
}

# ---------------------------------------------------------------- chunks
ch = pd.read_parquet(BASELINE.parent.parent / "bertopic" / "chunks_input.parquet")
stats["chunks"] = {
    "n_chunks": int(len(ch)),
    "episodes_chunked": int(ch["episode_id"].nunique()),
    "word_count_mean": round(float(ch["word_count"].mean()), 1),
    "word_count_median": int(ch["word_count"].median()),
    "word_count_p95": int(ch["word_count"].quantile(0.95)),
    "gender": {k: int(v) for k, v in ch["gender"].value_counts().items()},
}
g = ch["gender"].value_counts()
gtot = int(g.sum())
stats["chunks"]["gender_pct"] = {k: round(100 * int(v) / gtot, 1) for k, v in g.items()}

# ---------------------------------------------------------------- runs
cmp_path = ROOT / "outputs/bertopic_run_comparison_full_only.csv"
cmp = pd.read_csv(cmp_path)
cmp = cmp.sort_values("outlier_rate").reset_index(drop=True)
stats["runs"] = cmp[
    ["run_name", "topics_excluding_outlier", "outlier_rate", "assigned_count"]
].to_dict(orient="records")

# baseline topic_info
ti = pd.read_parquet(BASELINE / "topic_info.parquet")
ti_no = ti[ti["Topic"] != -1].copy()
stats["baseline"] = {
    "run_dir": str(BASELINE),
    "topics_including_outlier": int(len(ti)),
    "topics_excluding_outlier": int(len(ti_no)),
    "outlier_count": int(ti.loc[ti["Topic"] == -1, "Count"].iloc[0]),
    "largest_topic_count": int(ti_no["Count"].max()),
    "median_topic_size": int(ti_no["Count"].median()),
}

# =============================================================== FIGURES

# 1. chunk word-count distribution
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(ch["word_count"].clip(upper=350), bins=60, color="#3b6ea5")
ax.axvline(220, color="#c0392b", ls="--", lw=1, label="target 220")
ax.axvline(320, color="#7d3c98", ls=":", lw=1, label="max 320")
ax.set_xlabel("Words per chunk")
ax.set_ylabel("Number of chunks")
ax.set_title("Chunk length distribution (n=%d)" % len(ch))
ax.legend()
fig.savefig(FIG / "fig_chunk_wordcount.png")
plt.close(fig)

# 2. gender distribution (chunk level)
fig, ax = plt.subplots(figsize=(5, 3.5))
order = ["male", "female", "borderline", "unknown"]
vals = [int(ch["gender"].value_counts().get(k, 0)) for k in order]
colors = ["#2e86c1", "#c0392b", "#f39c12", "#7f8c8d"]
ax.bar(order, vals, color=colors)
for i, v in enumerate(vals):
    ax.text(i, v, f"{100*v/sum(vals):.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of chunks")
ax.set_title("F0-based vocal-gender distribution (chunk level)")
fig.savefig(FIG / "fig_gender_distribution.png")
plt.close(fig)

# 3. language distribution (episode level, top 8)
fig, ax = plt.subplots(figsize=(5.5, 3.5))
lang = ep["whisper_language"].value_counts().head(8)
ax.bar(lang.index.astype(str), lang.values, color="#27ae60")
ax.set_ylabel("Number of episodes")
ax.set_title("Detected language per episode (Whisper, top 8)")
fig.savefig(FIG / "fig_language_distribution.png")
plt.close(fig)

# 4. outlier rate by run
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(cmp["run_name"], cmp["outlier_rate"] * 100, color="#8e44ad")
for i, v in enumerate(cmp["outlier_rate"] * 100):
    ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Outlier (topic -1) rate (%)")
ax.set_title("Outlier rate across BERTopic runs")
ax.invert_yaxis()
fig.savefig(FIG / "fig_outlier_rate_by_run.png")
plt.close(fig)

# 5. topics excluding outlier by run
fig, ax = plt.subplots(figsize=(7, 3.8))
c2 = cmp.sort_values("topics_excluding_outlier")
ax.barh(c2["run_name"], c2["topics_excluding_outlier"], color="#16a085")
for i, v in enumerate(c2["topics_excluding_outlier"]):
    ax.text(v, i, f" {int(v)}", va="center", fontsize=8)
ax.set_xlabel("Number of topics (excluding outlier)")
ax.set_title("Topic count across BERTopic runs")
ax.invert_yaxis()
fig.savefig(FIG / "fig_topic_count_by_run.png")
plt.close(fig)

# 6. baseline top-20 non-outlier topics
fig, ax = plt.subplots(figsize=(7, 5))
top = ti_no.sort_values("Count", ascending=False).head(20).iloc[::-1]
labels = [n[:42] for n in top["Name"].astype(str)]
ax.barh(labels, top["Count"], color="#2c3e50")
ax.set_xlabel("Chunks assigned")
ax.set_title("Baseline run: 20 largest topics (excl. outlier)")
fig.savefig(FIG / "fig_baseline_top_topics.png")
plt.close(fig)

# 7. baseline topic-size distribution (log)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(ti_no["Count"], bins=50, color="#d35400")
ax.set_yscale("log")
ax.set_xlabel("Topic size (chunks)")
ax.set_ylabel("Number of topics (log)")
ax.set_title("Baseline run: topic-size distribution")
fig.savefig(FIG / "fig_baseline_topic_size.png")
plt.close(fig)

# 8. speakers per episode
fig, ax = plt.subplots(figsize=(5.5, 3.5))
sp = ep["n_speakers"].clip(upper=12).value_counts().sort_index()
ax.bar(sp.index.astype(int).astype(str), sp.values, color="#2980b9")
ax.set_xlabel("Diarized speakers per episode (capped at 12)")
ax.set_ylabel("Number of episodes")
ax.set_title("Speakers per episode (pyannote)")
fig.savefig(FIG / "fig_speakers_per_episode.png")
plt.close(fig)

(OUT / "corpus_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(json.dumps(stats, indent=2, ensure_ascii=False))
print("\nFigures written:", sorted(p.name for p in FIG.glob("*.png")))
