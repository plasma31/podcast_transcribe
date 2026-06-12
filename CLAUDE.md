# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research pipeline (German-language podcast thesis) that takes podcast audio and produces topic-modeled, speaker- and gender-attributed transcript data. There is no application or test suite — it is a set of CLI scripts run as batch jobs over a large local audio corpus. Audio, parquet, logs, and model artifacts are intentionally untracked (see `.gitignore`); only the scripts are versioned.

The work flows in three stages:

1. **Acquisition** — download episodes into `fyyd_downloads/<podcast_name>/` from the fyyd API, Podigee, or RSS feeds.
2. **Transcription + diarization + gender** — Whisper transcribes, pyannote diarizes, F0 analysis estimates perceived vocal gender per speaker. Output is per-episode parquet.
3. **Topic modeling** — segments are merged into word-count-bounded chunks and clustered with BERTopic.

## Environment setup

Two virtualenvs exist (both Python 3.12.3); activate the right one per stage:

- **`.venv`** — full stack: Whisper, pyannote, librosa, BERTopic. Use for stages 1–2 and combined runs.
- **`.venv_bertopic`** — BERTopic only. Use for stage 3 when running topic modeling in isolation.

Run scripts via the venv interpreter directly, e.g. `.venv/bin/python <script>`.
Direct Stage 1-2 dependencies are pinned in `requirements.base.txt`; install a
platform-appropriate PyTorch 2.8.0 family first.

Two hard runtime requirements:

- **`ffmpeg` must be on PATH** — Whisper/librosa decoding fails without it; the pipeline scripts `sys.exit` if it is missing.
- **A Hugging Face token must be exported** for pyannote and gated Whisper models. Scripts read `PYANNOTE_TOKEN`, then `HF_TOKEN`, then `HUGGINGFACE_TOKEN`, and raise if none is set. Never hardcode it (a token leak is already in the git history — commit `a0a6c18` "Token Removed").

GPU is auto-detected (`torch.cuda.is_available()`). Diarization stays on CPU unless `--diar_gpu` is passed. `d2c5a99` added a CUDA sanity check for the Linux environment.

## Running the pipeline

The canonical transcription and topic-modeling code lives in `pipeline/`. Acquisition utilities live in `acquisition/`, and their source spreadsheets/CSV files live in `data_sources/`.

**Import layout:** the pipeline entry points use sibling imports. Running them
as files from the repository root works because Python adds `pipeline/` to
`sys.path`; running them from inside `pipeline/` also works.

### Stage 2 — batch transcription/diarization/gender (resumable)

```bash
cd pipeline
../.venv/bin/python batch_podcast_runner.py \
  --downloads ../fyyd_downloads \
  --out_root ../outputs \
  --state_dir ../outputs/state \
  --whisper_model small \
  --gender \
  --diar_gpu \
  --limit 500 --retry_failed --rebuild_manifest
```

### Stage 3 — BERTopic (resumable chunk build + training)

```bash
cd pipeline
# Daily incremental chunk build only:
../.venv_bertopic/bin/python run_bertopic_from_manifest.py \
  --manifest ../outputs/state/manifest.parquet \
  --output-dir ../outputs/bertopic \
  --chunk-episode-limit 600 --no-train
# Train once enough chunks accumulated (add --force-train to retrain after param changes):
../.venv_bertopic/bin/python run_bertopic_from_manifest.py \
  --manifest ../outputs/state/manifest.parquet \
  --output-dir ../outputs/bertopic --train
```

`bertopic_typisierung.py` is the standalone CSV-input variant of the same modeling logic and the source of the shared helpers (`build_model`, `build_stopwords`, name-stopword loaders) that the manifest runner imports.

### Supporting scripts

- `acquisition/fyyd_download.py` and `acquisition/rss_download.py` — download episodes into `fyyd_downloads/` from fyyd or RSS.
- `acquisition/podigee_scrape.py` — collect Podigee episode URLs into `data_sources/podigee_episodes.csv`.
- `data_sources/list.xlsx` and `data_sources/redownload_list.xlsx` — acquisition source lists.
- `tools/audit_missing_speaker_gender.py` — audits episode parquet for missing/unknown speaker gender.
- `pipeline/reassign_bertopic_outliers.py` / `..._embeddings.py` — reassign `topic == -1` outlier chunks using a trained model (c-tf-idf / embeddings strategies).
- `pipeline/compare_bertopic_runs.py` — compare multiple `outputs/bertopic*/podcast_chunks_sw-de/` runs.
- `tools/report_directory_usage.py` — recursive directory file-count and size reporter.

## Architecture and data model

### `PodcastPipeline` (pipeline/pipeline_core.py)

The core class loads three models and exposes `process_episode()`, which returns an `EpisodeArtifacts` dataclass (episode record + segment records + debug payload). The flow per episode:

1. **Whisper** `transcribe()` → time-stamped text segments (`fp16=False` for stability).
2. **pyannote** `diarize()` (model `pyannote/speaker-diarization-3.1`) → speaker turns. Audio is loaded mono/16k via librosa. `_load_pyannote` tries `token` / `use_auth_token` / `hf_token` kwargs in turn to tolerate pyannote API drift.
3. **`match_segments()`** assigns each Whisper segment to the diarized speaker with maximum time overlap.
4. **Gender** — `estimate_speaker_gender()` concatenates each speaker's turns (up to `max_speaker_sec`) and calls `estimate_gender_from_f0()`. Gender is derived from **median fundamental frequency via `librosa.pyin`**, not a neural classifier — thresholds are `f0 < 155 Hz → male`, `> 185 Hz → female`, else `borderline`/`unknown`.

`episode_id = sha1(resolved absolute episode path)` — stable across runs as long as files are not moved.

### Output layout (under `--out_root`, conventionally `outputs/`)

- `parquet/episodes/<episode_id>.parquet` — one row per episode (language, counts, `speaker_gender_json`, runtime).
- `parquet/segments/<episode_id>.parquet` — one row per transcript segment (speaker, gender, f0, text, times).
- `json_debug/<episode_id>.json` — full debug payload (raw Whisper + diarized + matched segments).
- `state/manifest.parquet` — the job ledger; `state/failures.parquet` — failure log.

### Resumability via the manifest (StateStore)

`batch_podcast_runner.py` is built to be re-run safely. It maintains `manifest.parquet` with a `status` per episode (`pending` → `running` → `done`/`failed`). On each run it:

- rescans the downloads tree into an inventory and merges with the existing manifest (`--rebuild_manifest`), preserving prior status by `episode_id`;
- with `--skip_existing_outputs`, marks episodes `done` if their parquet outputs already exist;
- selects `pending` (plus `failed` with `--retry_failed`, plus `running`) up to `--limit`, processes them, and **saves the manifest after every episode** so an interrupted run loses at most one episode.

The BERTopic runner mirrors this: chunk building is episode-resumable (tracked in `chunk_build_state.parquet`, skips already-chunked `episode_id`s), and training is gated by a `_TRAINING_COMPLETE.json` marker unless `--force-train`.

### Chunking and topic modeling

`run_bertopic_from_manifest.py` reads the manifest, joins each episode's segment + episode parquet, and merges consecutive segments into chunks bounded by word count (`--chunk-target-words` 220, `--chunk-max-words` 320) and kept speaker-consistent by default (a chunk flushes on speaker change). Chunks with mixed speakers/genders are labeled `"mixed"`. BERTopic uses a multilingual SentenceTransformer (`paraphrase-multilingual-MiniLM-L12-v2` by default) + UMAP + HDBSCAN + `CountVectorizer`.

German stopwords come from `stopwords-iso`, augmented with podcast-specific terms and — importantly — **person-name stopwords** loaded from the `names-dataset` library (top-N first/last names per country, default `DE,AT,CH,TR,...`) so speaker names don't surface as topics. `pipeline/bertopic_extra_stopwords.txt` holds additional manual stopwords. `--names-dataset-mode all` exists but can produce hundreds of thousands of stopwords and is slow.

## Conventions

- Python is formatted with **Black** (`.vscode` sets `ms-python.black-formatter`); other files use Prettier.
- Each pipeline script sets `HF_HUB_DISABLE_SYMLINKS=1` at import time — required on mounted filesystems; keep it when adding new entrypoints.
- `outputs/bertopic_*` directories are parallel experiment runs (different embedding models / params encoded in the folder name, e.g. `bertopic_e5_mcs50_ms1`); don't overwrite them — pass a fresh `--output-dir` per experiment.
