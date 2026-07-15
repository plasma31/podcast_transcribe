# Chapter: Topic Modelling

## 1. Scope of this chapter

This chapter begins after Stage 3 has produced the canonical chunk corpus:

```text
outputs/common_chunks/chunks_input.parquet
```

The purpose is to explain exactly what BERTopic receives, what each modelling component contributes, why the parameters matter, which files are written, and how a downstream consumer should interpret the results.

The modelling sequence is:

```text
chunk_text
  -> SentenceTransformer embedding
  -> UMAP dimensionality reduction
  -> HDBSCAN clustering
  -> CountVectorizer and c-TF-IDF topic representation
  -> document-topic and topic-summary tables
```

The raw audio, Whisper segments, Stage 2 manifest, and episode Parquet files are not passed directly into the embedding model. They are used earlier to construct the chunk rows.

## 2. The modelling document

### 2.1 Why Whisper segments are not used directly

Whisper segments are created for speech recognition and timestamp alignment. They are frequently short, incomplete, and dependent on surrounding speech. A segment can contain only a phrase, backchannel, or partial sentence.

Embedding every segment separately would create many weak semantic vectors and would overrepresent brief conversational fragments.

### 2.2 Why full episodes are not used directly

A podcast episode can contain several topics, speakers, introductions, advertisements, and transitions. Representing a complete episode with one vector would require the clustering step to assign one dominant topic to material that is often topically mixed.

### 2.3 Why chunks are used

Chunks provide a middle level between the two extremes.

A chunk contains consecutive transcript segments from one episode and normally one diarized speaker. It is long enough to carry topical information but short enough to preserve local changes in discussion.

The main modelling table is:

```text
outputs/common_chunks/chunks_input.parquet
```

The main modelling column is:

```text
chunk_text
```

Each row is treated as one BERTopic document.

### 2.4 Meaning of the filename

`chunks_input.parquet` means **input to the embedding and topic-modelling process**.

It should not be interpreted as the input to the complete podcast pipeline. The complete pipeline starts with source lists and audio. The chunk file appears near the end of data preparation.

A clearer conceptual label is:

```text
canonical BERTopic document corpus
```

The physical filename remains unchanged because it is used by the existing scripts.

### 2.5 Chunk metadata retained beside the text

The embedding model only needs `chunk_text`, but the table preserves additional columns:

- `chunk_id` for stable joins;
- `episode_id` for episode provenance;
- `podcast_folder` and `episode_path` for source tracing;
- `speaker` and `gender` for subgroup analysis;
- `start` and `end` for locating the text in the episode;
- `word_count` and `source_segment_count` for quality and length checks.

This separation is important. Metadata is retained for interpretation and joins, but it does not automatically become part of the semantic embedding.

## 3. Embedding with SentenceTransformer

### 3.1 What an embedding represents

SentenceTransformer converts each chunk into a fixed-length numeric vector. Chunks with similar language and meaning should occupy nearby positions in the embedding space.

The vector is not a topic label. It is a machine-readable semantic representation used by the later dimensionality-reduction and clustering steps.

### 3.2 Default model

The default model in `run_bertopic_from_manifest.py` is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

It was selected as the practical baseline because the corpus is predominantly German, the model supports multiple languages, and it is computationally light enough for repeated experiments over approximately 191,000 documents.

Larger models were tested separately. A larger parameter count does not guarantee a better topic model because the final result also depends on how the embedding geometry interacts with UMAP and HDBSCAN.

### 3.3 Device selection

`--embedding-device` accepts:

- `auto`;
- `cpu`;
- `cuda`.

GPU execution is substantially more practical for the full corpus. CPU remains useful for compatibility, small tests, or systems without a supported CUDA environment.

### 3.4 Embedding cache

The grid-search workflow can store and reuse embedding matrices under:

```text
outputs/common_chunks/embedding_cache/
```

This avoids recomputing the most expensive shared input when only UMAP or HDBSCAN parameters change.

A cache is valid only for the exact ordered chunk table and embedding configuration used to create it. Row count alone is not sufficient provenance. The source chunk checksum, model name, filtering settings, sampling settings, and row order should be preserved with the cache metadata.

## 4. UMAP dimensionality reduction

### 4.1 Why dimensionality is reduced

SentenceTransformer embeddings contain many dimensions. HDBSCAN can operate more effectively and efficiently after the vectors are projected into a smaller space that preserves relevant neighbourhood structure.

UMAP performs this projection.

### 4.2 Main parameters

| Parameter      | Baseline | Interpretation                                                          |
| -------------- | -------: | ----------------------------------------------------------------------- |
| `n_neighbors`  |       30 | Size of the local neighbourhood used when learning the reduced geometry |
| `n_components` |        5 | Number of dimensions supplied to HDBSCAN                                |
| `min_dist`     |      0.0 | Allows nearby points to be packed tightly in the reduced space          |
| `metric`       |   cosine | Measures similarity in the original embedding space                     |
| `random_state` |       42 | Fixes the stochastic projection for reproducibility                     |

### 4.3 Why `n_neighbors` matters

A smaller value gives more weight to local structure and can produce finer clusters. A larger value incorporates broader neighbourhood information and can encourage larger, more globally stable groupings.

The parameter does not directly specify the number of topics. It changes the geometry on which HDBSCAN later operates.

### 4.4 Why a fixed random state is used

UMAP is stochastic. Without a fixed seed, repeated executions with identical data and parameters can produce different reduced coordinates and therefore different clusters.

A fixed random state does not make every library and hardware operation perfectly deterministic, but it removes a major source of avoidable variation.

## 5. HDBSCAN clustering

### 5.1 What HDBSCAN does

HDBSCAN searches the reduced vectors for dense regions. Each dense region becomes a candidate topic cluster.

Documents that do not belong confidently to a sufficiently dense region receive:

```text
topic = -1
```

The outlier class is an explicit model result, not a missing value.

### 5.2 Main parameters

| Parameter                  |  Baseline | Interpretation                                     |
| -------------------------- | --------: | -------------------------------------------------- |
| `min_cluster_size`         |        50 | Minimum size of a reported dense cluster           |
| `min_samples`              |         1 | Controls how conservative density membership is    |
| `metric`                   | euclidean | Distance measure in the reduced UMAP space         |
| `cluster_selection_method` |     `eom` | Selects stable clusters from the density hierarchy |

### 5.3 Why `min_cluster_size` matters

Lower values permit more small clusters. This increases topic granularity but can create fragmented or highly specific topics.

Higher values require more supporting documents. This can produce fewer and broader topics, while moving small thematic groups into larger clusters or the outlier class.

The parameter therefore expresses a substantive decision about the minimum amount of corpus evidence required before a theme is reported as a topic.

### 5.4 Why `min_samples` matters

A higher value makes HDBSCAN more conservative. Borderline documents are more likely to become outliers.

A low value of 1 was used in the baseline to avoid unnecessarily rejecting conversational documents from an already heterogeneous corpus.

### 5.5 Why outliers are retained

Podcast speech contains greetings, transitions, jokes, advertisements, personal exchanges, and fragments that may not form a stable corpus-level theme.

Forcing every chunk into a topic would improve numeric coverage but could reduce topic purity. The outlier rate must therefore be interpreted together with topic coherence and representative documents.

A lower outlier rate is not automatically evidence of a better model.

## 6. Topic representation with CountVectorizer and c-TF-IDF

### 6.1 Why clustering alone is insufficient

HDBSCAN produces numeric cluster assignments. It does not explain what the clusters mean.

BERTopic groups all documents assigned to a topic and uses class-based TF-IDF to identify words and phrases that are especially characteristic of that topic compared with the other topics.

### 6.2 Vectorizer parameters

| Parameter     | Baseline | Interpretation                                               |
| ------------- | -------: | ------------------------------------------------------------ |
| `ngram_range` |   1 to 3 | Allow single words, two-word phrases, and three-word phrases |
| `min_df`      |       10 | A term must appear in at least ten documents                 |
| `max_df`      |     0.95 | Terms present in more than 95% of documents are excluded     |

The n-gram range allows multi-word expressions to appear in a topic representation. This is useful when meaning is carried by a phrase rather than an isolated token.

### 6.3 Stopwords and person names

Conversational transcripts contain frequent function words, address terms, fillers, host names, and guest names. Without filtering, these tokens can dominate the topic labels even when they do not describe the substantive theme.

The pipeline can combine:

- German stopwords;
- manually curated conversational and corpus-specific stopwords;
- an optional person-name list;
- a local file such as `pipeline/bertopic_extra_stopwords.txt`.

Name removal must be applied carefully. A name can be noise when it identifies a recurring host, but it can also be substantively important when the research question concerns a public person or organisation.

The final stopword regime must therefore be recorded in `run_config.json` and treated as part of the model specification.

## 7. Optional topic reduction

`--nr-topics` can merge the discovered topics to a target count after the initial model has been fitted.

This step changes granularity. It does not replace the embedding or HDBSCAN stages and does not directly eliminate outliers.

A fixed target can make a result easier to report or compare, but it also introduces a researcher-selected level of aggregation. The unreduced model remains useful for inspecting the data-driven cluster structure.

## 8. Main runner and grid-search runner

### 8.1 `run_bertopic_from_manifest.py`

This is the canonical Stage 3 runner. It can:

- read the Stage 2 manifest;
- construct or resume chunks;
- train one BERTopic configuration;
- write model tables, visualisations, and a saved model.

It writes runner-local chunk files to the selected `--output-dir` and model files to:

```text
<output-dir>/podcast_chunks_sw-de/
```

### 8.2 `greedy_grid_search_bertopic_from_chunks.py`

This script starts after chunk construction. It does not download, transcribe, diarize, or rebuild chunks.

Its preferred input is:

```text
outputs/common_chunks/chunks_input.parquet
```

It evaluates combinations of:

- UMAP `n_neighbors`;
- HDBSCAN `min_cluster_size`;
- the fixed supporting parameters supplied through the command line.

It can reuse a common embedding matrix so that parameter comparisons are based on the same document vectors.

### 8.3 Why the common chunk corpus is necessary

Model comparisons are meaningful only when the document population is held constant.

If each run rebuilt chunks independently, a difference in topic count or outlier rate could be caused by different document boundaries rather than the embedding or clustering parameters.

The canonical common file therefore defines the experiment population:

```text
outputs/common_chunks/chunks_input.parquet
```

Each run should record its source checksum or provenance manifest.

## 9. Model output files

A completed run writes the following files under `podcast_chunks_sw-de/`.

| File                          | Main contents                                           | Reader question answered                                                         |
| ----------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `doc_topics.parquet`          | One row per chunk with its assigned topic               | Which topic was assigned to this document?                                       |
| `chunks_with_topics.parquet`  | Full chunk table plus `doc_id` and `topic`              | What source metadata belongs to each assignment?                                 |
| `topic_info.parquet`          | Topic identifier, count, name, representation, examples | Which topics exist and how large are they?                                       |
| `topic_words.parquet`         | Top terms and c-TF-IDF scores                           | Which words distinguish this topic?                                              |
| `representative_docs.parquet` | Example chunks for each topic                           | What does the topic look like in context?                                        |
| `doc_topic_probs.parquet`     | Optional probability matrix                             | How strongly does each document relate to topics when probabilities are enabled? |
| `bertopic_model/`             | Saved model artefacts                                   | How can the trained model be reloaded?                                           |
| `run_config.json`             | Parameters, paths, runtime, counts                      | How was this result produced?                                                    |
| `_TRAINING_COMPLETE.json`     | Completion marker                                       | Has this run already finished?                                                   |
| `topics_overview.html`        | Intertopic map                                          | Which topics are close or distant?                                               |
| `topics_barchart.html`        | Topic-term score charts                                 | Which terms support each label?                                                  |
| `topics_hierarchy.html`       | Topic dendrogram                                        | Which topics merge at low or high distance?                                      |

### 9.1 `doc_topics.parquet`

This is the most useful application-facing result table.

Important columns:

| Column              | Meaning                                            |
| ------------------- | -------------------------------------------------- |
| `doc_id`            | Row identifier assigned during model fitting       |
| `chunk_id`          | Stable join key back to the canonical chunk corpus |
| `episode_id`        | Source episode                                     |
| `speaker`, `gender` | Retained chunk metadata                            |
| `start`, `end`      | Source time range                                  |
| `topic`             | Topic identifier or `-1`                           |
| `chunk_text`        | Text that was embedded and clustered               |

`doc_id` is run-specific. `chunk_id` is the safer cross-run key.

### 9.2 `topic_info.parquet`

This table describes the topic inventory and includes the outlier row when topic `-1` exists.

The generated topic name is an automatic representation based on high-scoring terms. It should be treated as evidence for interpretation, not as a final human label.

### 9.3 `topic_words.parquet`

This table preserves the terms and their scores. It is more informative than a single concatenated topic name because it allows the researcher to inspect the relative evidence behind the label.

### 9.4 `representative_docs.parquet`

Representative chunks are necessary for substantive validation. A topic cannot be assessed reliably from top words alone, especially when words are polysemous or transcript errors are present.

## 10. Reading the diagnostic visualisations

### 10.1 Intertopic distance map

`topics_overview.html` displays topics as circles in a two-dimensional projection.

- circle size represents topic frequency;
- proximity suggests similarity between topic representations;
- overlap or crowding can indicate closely related topics;
- isolated circles suggest more distinct representations.

The two-dimensional layout is a diagnostic projection. Exact distances should not be interpreted as a direct physical scale.

### 10.2 Topic-word bar charts

`topics_barchart.html` shows the highest c-TF-IDF terms for selected topics.

The chart answers: which words make this topic different from other topics?

Terms such as different grammatical forms of the same word can appear separately because the vectorizer does not automatically reduce every German word to a common lemma. This is a representation choice, not evidence that the model considers the forms unrelated in every semantic sense.

### 10.3 Hierarchical topic view

`topics_hierarchy.html` builds a dendrogram from topic representations.

- topics joined at a low distance are relatively similar;
- topics joined at a high distance are more distinct;
- a branch indicates a possible broader family of topics;
- the diagram can support a decision about post-hoc merging.

The hierarchy is not a manually defined taxonomy. It is derived from similarity among the model's topic representations.

## 11. How to judge a model

There is no labelled gold-standard topic assignment for this corpus. Evaluation must combine several forms of evidence.

| Criterion                | Question                                                               |
| ------------------------ | ---------------------------------------------------------------------- |
| Coverage                 | What proportion of chunks receives a non-outlier topic?                |
| Granularity              | Are there too few broad topics or too many fragmented topics?          |
| Coherence                | Do the high-scoring words belong together?                             |
| Representative documents | Do actual chunks support the topic interpretation?                     |
| Stability                | Do repeated or nearby configurations recover similar themes?           |
| Usefulness               | Does the topic resolution answer the thesis and application questions? |
| Runtime                  | Is the configuration practical to reproduce and extend?                |

A selected model should therefore be described as the most defensible compromise for the research purpose, not as an objectively best model.

## 12. Outlier reassignment

The repository contains optional scripts for reassigning topic `-1` documents.

Reassignment can increase coverage, but it changes the interpretation of the model by forcing previously uncertain documents toward existing topics.

The original assignments must be preserved. Reassigned outputs should use separate filenames and should be presented as a sensitivity analysis rather than silently replacing the HDBSCAN result.

## 13. Reproducibility requirements

A reported topic-model result should identify:

- the canonical chunk file and checksum;
- the number of chunk rows;
- any filtering or sampling step;
- the embedding model and revision;
- UMAP parameters;
- HDBSCAN parameters;
- vectorizer and stopword settings;
- optional topic-reduction target;
- whether probabilities were calculated;
- the output run directory;
- the resulting topic and outlier counts.

These values are persisted in `run_config.json` and should be treated as part of the result, not as incidental implementation details.
