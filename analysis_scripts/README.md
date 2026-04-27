## analysis_scripts

### assign_cluster.py

Assign clusters to query genomes using k-nearest neighbors classification. Generates embeddings for prompt and query genomes, performs KNN classification with configurable neighbor count, and evaluates classification accuracy using multiple metrics.

```
python assign_cluster.py \
  --model_path model.chk \
  --tokenizer_path tokenizer.json \
  --prompt_file /path/to/prompt/genomes.txt \
  --prompt_labels /path/to/prompt_labels.csv \
  --query_file /path/to/query/genomes.txt \
  --query_labels /path/to/query_labels.csv \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 8 \
  --max_seq_length 8192 \
  --attention_window 512 \
  --batch_size 8 \
  --pooling mean \
  --n_neighbors 5 \
  --outpref output_clusters
```

Alternatively, previously computed embeddings can be supplied directly via `--prompt_embeddings` and `--query_embeddings` to skip re-embedding. If `--query_labels` contains a second column with true cluster assignments, classification accuracy metrics are reported.

Key arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model_path` | Yes | — | Path to model checkpoint |
| `--tokenizer_path` | Yes | — | Path to tokenizer JSON file |
| `--prompt_file` | One of | — | Genome file for reference embeddings |
| `--prompt_embeddings` | One of | — | Pre-computed reference embeddings file |
| `--prompt_labels` | Yes | — | CSV of genome names (col 1) and optional cluster assignments (col 2) for prompt genomes |
| `--query_file` | One of | — | Genome file for query embeddings |
| `--query_embeddings` | One of | — | Pre-computed query embeddings file |
| `--query_labels` | No | None | CSV of genome names (col 1) and optional cluster assignments (col 2) for query genomes |
| `--n_neighbors` | No | 5 | Number of neighbors for KNN classification |
| `--pooling` | No | mean | Pooling strategy (`mean` or `max`) |
| `--outpref` | No | simulated_genomes | Output file prefix |
| `--DDP` | No | False | Enable multi-GPU inference via DDP |
| `--ignore_unknown` | No | False | Ignore unknown tokens |
| `--randomise` | No | False | Randomise input sequences |

---

### compute_SHAP.py

Compute SHAP (SHapley Additive exPlanations) values for model interpretability. Analyzes feature importance by calculating SHAP values for specific target tokens, providing insights into model decision-making processes and token relationships.

```
python compute_SHAP.py \
  --model_path model.chk \
  --tokenizer_path tokenizer.json \
  --prompt_file /path/to/genomes.txt \
  --target-token TARGET_TOKEN \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 8 \
  --max_seq_length 8192 \
  --attention_window 512 \
  --batch_size 8 \
  --outpref output_shap
```

Key arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model_path` | Yes | — | Path to model checkpoint |
| `--tokenizer_path` | Yes | — | Path to tokenizer JSON file |
| `--prompt_file` | Yes | — | Tab-separated file with genome IDs and sequences |
| `--target-token` | Yes | — | Token to compute SHAP values for |
| `--outpref` | No | simulated_genomes | Output file prefix |
| `--encoder_only` | No | False | Use encoder input only |
| `--seed` | No | 42 | Random seed |
| `--DDP` | No | False | Enable multi-GPU inference via DDP |
| `--randomise` | No | False | Randomise input tokens |

---

### compute_cosine.py

Calculate cosine similarity matrices between genome embeddings. Generates pairwise cosine similarity scores for all genome combinations in the dataset, enabling similarity-based clustering and relationship analysis.

```
python compute_cosine.py \
  --model_path model.chk \
  --tokenizer_path tokenizer.json \
  --prompt_file /path/to/genomes.txt \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 8 \
  --max_seq_length 8192 \
  --attention_window 512 \
  --batch_size 8 \
  --outpref output_cosine
```

Key arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model_path` | Yes | — | Path to model checkpoint |
| `--tokenizer_path` | Yes | — | Path to tokenizer JSON file |
| `--prompt_file` | Yes | — | Path to genome sequences file |
| `--outpref` | No | simulated_genomes | Output file prefix |
| `--encoder_only` | No | False | Use encoder input only |
| `--DDP` | No | False | Enable multi-GPU inference via DDP |
| `--randomise` | No | False | Randomise input sequences |

---

### compute_sequence_embedding.py

Generate sequence embeddings from trained panBART model. Processes genome sequences through the model to produce fixed-dimensional embeddings suitable for downstream analysis like clustering and classification.

```
python compute_sequence_embedding.py \
  --model_path model.chk \
  --tokenizer_path tokenizer.json \
  --prompt_file /path/to/genomes.txt \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 8 \
  --max_seq_length 8192 \
  --attention_window 512 \
  --batch_size 8 \
  --pooling mean \
  --outpref output_embeddings
```

Key arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model_path` | Yes | — | Path to model checkpoint |
| `--tokenizer_path` | Yes | — | Path to tokenizer JSON file |
| `--prompt_file` | Yes | — | Path to genome sequences file |
| `--labels` | No | None | CSV of genome names (col 1) and optional cluster assignments (col 2) |
| `--pooling` | No | mean | Pooling strategy (`mean` or `max`) |
| `--outpref` | No | simulated_genomes | Output file prefix |
| `--encoder_only` | No | False | Use encoder input only |
| `--ignore_unknown` | No | False | Ignore unknown tokens |
| `--DDP` | No | False | Enable multi-GPU inference via DDP |
| `--randomise` | No | False | Randomise input sequences |

---

### compute_token_likelihood.py

Calculate token likelihoods and pseudolikelihood scores for genome sequences. Evaluates model predictions across genome positions, computes statistical metrics, and generates per-gene likelihood assessments.

```
python compute_token_likelihood.py \
  --model_path model.chk \
  --tokenizer_path tokenizer.json \
  --prompt_file /path/to/genomes.txt \
  --gene_list /path/to/gene_list.tsv \
  --reps_dict /path/to/reps_dict.pkl \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 8 \
  --max_seq_length 8192 \
  --attention_window 512 \
  --batch_size 8 \
  --outpref output_likelihoods
```

Key arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model_path` | Yes | — | Path to model checkpoint |
| `--tokenizer_path` | Yes | — | Path to tokenizer JSON file |
| `--prompt_file` | Yes | — | Path to genome sequences file |
| `--gene_list` | Yes | — | TSV file with representative gene IDs in the first column (e.g. Bakta `_full.tsv` output) |
| `--reps_dict` | Yes | — | PKL file mapping representative sequences to tokens (from `tokenise_clusters.py`) |
| `--outpref` | No | simulated_genomes | Output file prefix |
| `--encoder_only` | No | False | Use encoder input only |
| `--parse_gene_id` | No | False | Augment gene IDs to legacy format |
| `--max-SHAP` | No | False | Calculate SHAP values at the highest-pseudolikelihood position |
| `--DDP` | No | False | Enable multi-GPU inference via DDP |
| `--randomise` | No | False | Randomise input sequences |
| `--seed` | No | 42 | Random seed |

---

### generate_clusters.py

Generate Leiden clusters from sequence embeddings using scanpy. Performs community detection on embedding space with configurable parameters, optimizing clustering quality using Adjusted Rand Index and Adjusted Mutual Information scores.

```
python generate_clusters.py \
  --prompt_file /path/to/genomes.txt \
  --prompt_embeddings /path/to/embeddings.csv \
  --prompt_labels /path/to/labels.csv \
  --n_neighbors 10 \
  --resolution 1.0 \
  --outpref output_leiden
```

Key arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--prompt_file` | Yes | — | Path to genome sequences file |
| `--prompt_embeddings` | Yes | — | Pre-computed embeddings file (e.g. output of `compute_sequence_embedding.py`) |
| `--prompt_labels` | No | None | CSV of genome names (col 1) and optional true cluster assignments (col 2) |
| `--n_neighbors` | No | 10 | Number of neighbors for KNN graph construction |
| `--resolution` | No | 1.0 | Leiden clustering resolution |
| `--outpref` | No | simulated_genomes | Output file prefix |
| `--pooling` | No | mean | Pooling strategy (`mean` or `max`) |
| `--ignore_unknown` | No | False | Ignore unknown tokens |

---

### prompt_model.py

Generate genome sequences using trained panBART model through prompting. Accepts partial genome sequences as prompts and completes them using the trained model, with options for different sampling strategies and sequence randomization.

```
python prompt_model.py \
  --model_path model.chk \
  --tokenizer_path tokenizer.json \
  --prompt_file /path/to/genomes.txt \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 8 \
  --max_seq_length 8192 \
  --attention_window 512 \
  --batch_size 8 \
  --prop_masked 0.15 \
  --prop_prompt_kept 1.0 \
  --num_seq 1 \
  --outpref output_generated
```

Key arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model_path` | Yes | — | Path to model checkpoint |
| `--tokenizer_path` | Yes | — | Path to tokenizer JSON file |
| `--prompt_file` | Yes | — | Path to genome sequences file |
| `--temperature` | No | None | Sampling temperature (if unset, uses greedy decoding) |
| `--prop_masked` | No | 0.15 | Proportion of prompt to mask for the encoder |
| `--prop_prompt_kept` | No | 1.0 | Proportion of the prompt (from the start) to retain before encoding |
| `--num_seq` | No | 1 | Number of sequences to generate per prompt |
| `--k_size` | No | 3 | K-mer size for Jaccard distance calculation |
| `--shuffle_genomes` | No | False | Shuffle contig order in prompt |
| `--encoder_only` | No | False | Use encoder input only |
| `--generate` | No | False | Generate sequences iteratively rather than as a block |
| `--outpref` | No | simulated_genomes | Output file prefix |
| `--DDP` | No | False | Enable multi-GPU inference via DDP |

---

### pseudolikelihood.py

Calculate pseudolikelihood scores for genome sequences under the trained model. Evaluates how well the model predicts existing genome sequences, providing quantitative measures of model fit and sequence quality.

```
python pseudolikelihood.py \
  --model_path model.chk \
  --tokenizer_path tokenizer.json \
  --prompt_file /path/to/genomes.txt \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 8 \
  --max_seq_length 8192 \
  --attention_window 512 \
  --batch_size 8 \
  --outpref output_pseudolikelihood
```

Key arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model_path` | Yes | — | Path to model checkpoint |
| `--tokenizer_path` | Yes | — | Path to tokenizer JSON file |
| `--prompt_file` | Yes | — | Path to genome sequences file |
| `--per-gene` | No | False | Calculate per-gene pseudolikelihoods |
| `--ignore_unknown` | No | False | Ignore unknown tokens |
| `--outpref` | No | simulated_genomes | Output file prefix |
| `--encoder_only` | No | False | Use encoder input only |
| `--DDP` | No | False | Enable multi-GPU inference via DDP |
| `--randomise` | No | False | Randomise input sequences |

---

### test_model_accuracy.py

Evaluate model accuracy on test datasets. Loads trained model checkpoints and computes comprehensive performance metrics including loss, perplexity, accuracy, precision, recall, and F1 scores on held-out test data.

```
python test_model_accuracy.py \
  --test_file /path/to/test/genomes.txt \
  --model_save_path model.chk \
  --tokenizer_path tokenizer.json \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 8 \
  --max_seq_length 8192 \
  --attention_window 512 \
  --batch_size 8 \
  --num_workers 1
```

Key arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--test_file` | Yes | — | Path to test genome sequences file |
| `--model_save_path` | Yes | — | Path to model checkpoint |
| `--tokenizer_path` | Yes | — | Path to tokenizer JSON file |
| `--max_vocab_size` | No | None | Maximum vocabulary size; tokens beyond this are mapped to `<UNK>` |
| `--encoder_only` | No | False | Evaluate using encoder input only |
| `--gradient_checkpointing` | No | False | Enable gradient checkpointing (reduces memory use) |
| `--ignore_unknown` | No | False | Ignore unknown tokens during evaluation |
| `--DDP` | No | False | Enable multi-GPU evaluation via DDP |
| `--num_workers` | No | 1 | Number of data-loading threads |
