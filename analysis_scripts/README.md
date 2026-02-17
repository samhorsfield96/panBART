## analysis_scripts

### assign_cluster.py

Assign clusters to query genomes using k-nearest neighbors classification. Generates embeddings for prompt and query genomes, performs KNN classification with configurable neighbor count, and evaluates classification accuracy using multiple metrics.

### compute_SHAP.py

Compute SHAP (SHapley Additive exPlanations) values for model interpretability. Analyzes feature importance by calculating SHAP values for specific target tokens, providing insights into model decision-making processes and token relationships.

### compute_cosine.py

Calculate cosine similarity matrices between genome embeddings. Generates pairwise cosine similarity scores for all genome combinations in the dataset, enabling similarity-based clustering and relationship analysis.

### compute_sequence_embedding.py

Generate sequence embeddings from trained panBART model. Processes genome sequences through the model to produce fixed-dimensional embeddings suitable for downstream analysis like clustering and classification.

### compute_token_likelihood.py

Calculate token likelihoods and pseudolikelihood scores for genome sequences. Evaluates model predictions across genome positions, computes statistical metrics, and generates per-gene likelihood assessments.

### generate_clusters.py

Generate Leiden clusters from sequence embeddings using scanpy. Performs community detection on embedding space with configurable parameters, optimizing clustering quality using Adjusted Rand Index and Adjusted Mutual Information scores.

### prompt_model.py

Generate genome sequences using trained panBART model through prompting. Accepts partial genome sequences as prompts and completes them using the trained model, with options for different sampling strategies and sequence randomization.

### pseudolikelihood.py

Calculate pseudolikelihood scores for genome sequences under the trained model. Evaluates how well the model predicts existing genome sequences, providing quantitative measures of model fit and sequence quality.

### test_model_accuracy.py

Evaluate model accuracy on test datasets. Loads trained model checkpoints and computes comprehensive performance metrics including loss, perplexity, accuracy, precision, recall, and F1 scores on held-out test data.
