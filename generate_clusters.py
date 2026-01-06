import argparse
import pandas as pd
import numpy as np

from compute_sequence_embedding import read_prompt_file
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import normalize
import scanpy as sc

def parse_args_script():

    parser = argparse.ArgumentParser(description="Generate Leiden clusters from embeddings.")
    # additional analysis flags
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the text file containing the prompt.")
    parser.add_argument('--prompt_labels', required=False, default=None, help='csv file describing prompt_file genome names in first column. No header. Can have second column with assigned clusters.')
    parser.add_argument("--randomise", default=False, action="store_true", help="Randomise sequence for upon input.")
    parser.add_argument("--outpref", type=str, default="simulated_genomes", help="Output prefix for simulated genomes. Default = 'simulated_genomes'")
    parser.add_argument("--pooling", choices=['mean', 'max'], help="Pooling for embedding generation. Defaualt = 'mean'.")
    parser.add_argument("--ignore_unknown", default=False, action="store_true", help="Ignore unknown tokens during calculations.")
    parser.add_argument("--prompt_embeddings", required=True, type=str, help="Previously computed prompt embeddings.")
    parser.add_argument("--n_neighbors", default="10", help="Number of neighbors for KNN classification.")
    parser.add_argument("--resolution", default="1.0", help="Resolution for leiden clustering.")

    args = parser.parse_args()

    return args

def query_model(args, prompt_list, genome_labels, cluster_assignments):
    
    print("Reading embeddings...")
    prompt_list_df = pd.read_csv(args.prompt_embeddings, header=None, index_col=False)
    
    print("Generating training data...")
    # add known labels and use k-NN to generate labels for unknown
    X_train = prompt_list_df.iloc[:, 1:].values  # Features (N-dimensional embeddings)
    y_train = np.array(cluster_assignments)   # Labels

    X_train = normalize(X_train)

    n_neighbors_list = [int(k) for k in args.n_neighbors.split(",")]
    leiden_resolution_list = [float(j) for j in args.resolution.split(",")]

    per_iteration_accuracy = []

    print("Iterating through nearest neighbours...")
    for n_neighbors in n_neighbors_list:
        print(f"K: {n_neighbors}")
        for leiden_resolution in leiden_resolution_list:
            print(f"Leiden: {leiden_resolution}")

            adata = sc.AnnData(X_train)

            # generate KNN graph
            sc.tl.pca(adata)
            sc.pp.neighbors(
                adata,
                n_neighbors=n_neighbors,
                metric="cosine"
            )

            sc.tl.leiden(
                adata,
                resolution=leiden_resolution,
                key_added="leiden",
                flavor="igraph",
                n_iterations=2
            )

            leiden_labels = adata.obs["leiden"].astype(int).to_numpy()

            df_pred = pd.DataFrame(columns=['Taxon', 'predicted_label'])
            df_pred['Taxon'] = prompt_list_df.iloc[:, 0].values
            df_pred['predicted_label'] = leiden_labels
            df_pred.to_csv(args.outpref + f"_K_{n_neighbors}_resolution_{leiden_resolution}_predictions.tsv", sep='\t', index=False)

            if args.prompt_labels != None:

                true_labels = np.array(cluster_assignments)

                df_true = pd.DataFrame(columns=['Taxon', 'predicted_label', 'true_label'])
                df_true['Taxon'] = prompt_list_df.iloc[:, 0].values
                df_true['predicted_label'] = leiden_labels
                df_true['true_label'] = true_labels

                df_true.to_csv(args.outpref + f"_K_{n_neighbors}_resolution_{leiden_resolution}_true.tsv", sep='\t', index=False)

                ari = adjusted_rand_score(true_labels, leiden_labels)
                ami = adjusted_mutual_info_score(true_labels, leiden_labels)
                

                per_iteration_accuracy.append({
                    'K': n_neighbors,
                    'Leiden_resolution': leiden_resolution,
                    'ARI': ari,
                    'AMI': ami
                })
    

    if args.prompt_labels != None:
        per_label_df = pd.DataFrame(per_iteration_accuracy)

        # Save to TSV
        per_label_df.to_csv(args.outpref + f"_per_iter_accuracy.tsv", sep='\t', index=False)

def main():
    args = parse_args_script()

    print("Reading prompt...")
    genome_labels = []
    cluster_assignments = []
    if args.prompt_labels != None:
        with open(args.prompt_labels, "r") as i:
            i.readline()
            for line in i:
                split_line = line.rstrip().split(",")
                genome_name = split_line[0]
                genome_labels.append(genome_name)
                cluster_assignments.append(split_line[1])

        # parse prompt file and additional query files for assignment to clusters
        prompt_list, genome_labels = read_prompt_file(args.prompt_file, genome_labels)
    else:
        prompt_list, genome_labels = read_prompt_file(args.prompt_file)

    query_model(args, prompt_list, genome_labels, cluster_assignments)

if __name__ == "__main__":
    main()