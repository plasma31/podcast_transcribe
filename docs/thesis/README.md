# Thesis Documentation

Methodological documentation of the German-language podcast processing pipeline
and the first topic models, written as draft thesis-chapter prose. The corpus,
parameters, and figures are all reproducible from the committed pipeline
outputs.

## Contents

1. **[01_pipeline_and_data.md](01_pipeline_and_data.md)** — The three-stage
   pipeline (acquisition; transcription + diarization + vocal gender; topic
   modelling), the directory/output layout, full data dictionaries for every
   table, and the corpus statistics.
2. **[02_topic_modeling.md](02_topic_modeling.md)** — Chunking, the BERTopic
   pipeline and its baseline parameters with justifications, how the diagnostic
   diagrams work, and outlier reassignment.
3. **[03_results_and_figures.md](03_results_and_figures.md)** — The 8-run model
   comparison grid, outcomes, and the baseline topic structure.

## Reproducibility

| Artefact | How to regenerate |
|---|---|
| Figures (`figures/*.png`) + `corpus_stats.json` | `.venv/bin/python docs/thesis/_make_stats_and_figures.py` |
| Cross-run comparison CSV | `cd pipeline && ../.venv_bertopic/bin/python compare_bertopic_runs.py` |
| A topic-model run | `cd pipeline && ../.venv_bertopic/bin/python run_bertopic_from_manifest.py --manifest ../outputs/state/manifest.parquet --output-dir ../outputs/<name> --train` |

See the repository [`CLAUDE.md`](../../CLAUDE.md) for the full pipeline run
commands and environment setup.

## Key figures at a glance

- **Corpus:** 84 podcasts, 4,530 episodes (4,416 processed, 97.5 %),
  2,039,935 transcript segments, 191,183 chunks.
- **Language:** 97.6 % German.
- **Vocal gender (chunks):** 50.2 % male, 38.1 % female, 10.0 % borderline,
  1.7 % unknown.
- **Baseline model:** MiniLM + UMAP + HDBSCAN → 370 topics, 57.5 % outliers.
- **Best coverage:** `minilm_n100_t200` → 199 topics, 51.0 % outliers.

## References

The methods are documented in the following primary sources (author–year;
import into your reference manager):

- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based
  TF-IDF procedure.* arXiv:2203.05794.
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold
  Approximation and Projection for Dimension Reduction.* arXiv:1802.03426.
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). *Density-Based
  Clustering Based on Hierarchical Density Estimates.* PAKDD. (HDBSCAN.)
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks.* EMNLP. (Sentence-Transformers / SBERT.)
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I.
  (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.*
  arXiv:2212.04356. (Whisper.)
- Bredin, H., et al. (2020). *pyannote.audio: neural building blocks for speaker
  diarization.* ICASSP. (Diarization; pipeline `speaker-diarization-3.1`.)
- Wang, L., et al. (2024). *Multilingual E5 Text Embeddings.* arXiv:2402.05672.
  (`intfloat/multilingual-e5-large`.)
- Song, K., Tan, X., Qin, T., Lu, J., & Liu, T.-Y. (2020). *MPNet: Masked and
  Permuted Pre-training for Language Understanding.* NeurIPS.
  (`paraphrase-multilingual-mpnet-base-v2`.)
- McInnes, L., et al. *stopwords-iso* and *names-dataset* libraries (German
  stopwords and person-name suppression).
