# Thesis Documentation

This folder documents the podcast-processing pipeline from the perspective of a reader who did not implement it.

The purpose is not only to describe what the code intends to do. Each chapter explains the observable data flow:

1. what enters a step;
2. what transformation is applied;
3. which files are written;
4. where those files are stored;
5. whether they are committed to GitHub;
6. which later process or person consumes them;
7. why the implementation decision was made;
8. which limitations remain.

## Essential storage distinction

A fresh GitHub clone contains source code, source lists, dependency files, and documentation. It does **not** contain the full research corpus or model outputs.

The following directories are created during execution and are ignored by Git:

```text
fyyd_downloads/     downloaded podcast audio
outputs/            manifests, Parquet tables, chunks, embeddings, and model runs
logs/               execution logs
artifacts/           acquisition reports and generated support files
dist/               generated PDF and archive exports
```

Therefore, a documented path such as `outputs/common_chunks/chunks_input.parquet` is a runtime artefact on the processing machine or mounted project storage. It is not expected to appear in the GitHub file browser.

S3 paths in the documentation describe an optional shared-storage mirror. The current scripts do not upload to S3 automatically.

## Reading order

### 1. [Data Pipeline and Corpus](01_pipeline_and_data.md)

Read this first. It explains:

- the complete three-stage pipeline;
- the distinction between tracked and generated directories;
- the fyyd acquisition logic and its decisions;
- the Stage 2 manifest and transcript outputs;
- the exact Stage 3 chunk-construction algorithm;
- why `chunks_input.parquet` is called an input file;
- why the same filename appears in model-run and common-corpus directories;
- the three downstream handoff levels;
- the difference between local outputs and an optional S3 copy.

### 2. [Topic Modelling](02_topic_modeling.md)

This chapter begins where chunk construction ends. It explains:

- why a chunk, rather than a Whisper segment or full episode, is the modelling document;
- which column is passed to the embedding model;
- how SentenceTransformer, UMAP, HDBSCAN, CountVectorizer, and c-TF-IDF interact;
- what each model output means;
- how parameter choices change coverage, granularity, and interpretability;
- why topic `-1` is retained as an explicit result.

### 3. [Results and Model Comparison](03_results_and_figures.md)

This chapter explains:

- which runs are directly comparable;
- which input corpus they share;
- which parameters differ;
- what “better” means in an unsupervised setting;
- how to interpret topic count, outlier rate, coherence, runtime, and topic examples together;
- why the selected model is a reasoned compromise rather than an objectively best model.

## Data products at a glance

| Processing level | Main file | Meaning | Consumer |
|---|---|---|---|
| Stage 2 episode ledger | `outputs/state/manifest.parquet` | One row per discovered episode, including processing status and output paths | Batch runner, audit scripts, Stage 3 |
| Stage 2 transcript segments | `outputs/parquet/segments/<episode_id>.parquet` | Timestamped speaker-attributed ASR segments | Chunk builder or consumers needing raw transcript timing |
| Stage 3 canonical documents | `outputs/common_chunks/chunks_input.parquet` | One stable document row per chunk; `chunk_text` is the modelling text | Embedding models, BERTopic, search, external applications |
| BERTopic document assignments | `<selected-run>/podcast_chunks_sw-de/doc_topics.parquet` | One chunk plus its assigned topic | Thesis analysis and application topic browsing |
| BERTopic topic inventory | `<selected-run>/podcast_chunks_sw-de/topic_info.parquet` | Topic identifiers, counts, labels, and representations | Topic interpretation and reporting |

## Reproducibility commands

| Artefact | Regeneration command |
|---|---|
| Figures and `corpus_stats.json` | `.venv/bin/python docs/thesis/_make_stats_and_figures.py` |
| Cross-run comparison CSV | `cd pipeline && ../.venv_bertopic/bin/python compare_bertopic_runs.py` |
| Chunk build and topic-model run | `cd pipeline && ../.venv_bertopic/bin/python run_bertopic_from_manifest.py --manifest ../outputs/state/manifest.parquet --output-dir ../outputs/<run-name> --train` |
| Speaker and F0 audit | `.venv/bin/python tools/audit_missing_speaker_gender.py --manifest outputs/state/manifest.parquet` |
| Combined documentation PDF | `.venv/bin/python docs/thesis/_build_pdf.py` |

The PDF command writes `dist/thesis_documentation.pdf`. The `dist/` directory is generated and ignored by Git.

## Current corpus snapshot

The current documented snapshot contains:

- 84 podcasts in the Stage 2 manifest;
- 4,530 registered episodes;
- 4,416 successfully processed episodes;
- 2,039,935 transcript segments;
- 191,183 chunks from 4,400 episodes used in the full chunk corpus;
- 97.6% of successfully processed episodes detected as German.

These numbers describe a particular persisted corpus state. They should be updated when the manifest or canonical chunk corpus changes.

## Documentation maintenance rule

Whenever a pipeline change introduces a new file or changes a path, the documentation should state:

- whether the file is tracked or generated;
- the script and function that create it;
- the command-line option that determines its location;
- its schema or most important columns;
- whether it is canonical, a copy, a cache, or a model-specific result;
- the downstream consumer;
- whether an S3 version actually exists or is only recommended.
