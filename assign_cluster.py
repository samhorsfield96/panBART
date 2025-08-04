from parse_args import *

from compute_sequence_embedding import *
from sklearn.neighbors import KNeighborsClassifier

def parse_args_script():
    parser = parse_args()

    # additional analysis flags
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the text file containing the prompt.")
    parser.add_argument('--prompt_labels', required=True, help='csv file describing prompt_file genome names in first column. No header. Can have second column with assigned clusters.')
    parser.add_argument("--randomise", default=False, action="store_true", help="Randomise sequence for upon input.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the text file containing an additional prompt for querying.")
    parser.add_argument("--query_labels", type=str, default=None, required=False, help="csv file describing query_file genome names in first column. No header. Can have second column with assigned clusters.")
    parser.add_argument("--outpref", type=str, default="simulated_genomes", help="Output prefix for simulated genomes. Default = 'simulated_genomes'")
    parser.add_argument("--DDP", action="store_true", default=False, help="Multiple GPUs used via DDP during training.")
    parser.add_argument("--pooling", choices=['mean', 'max'], help="Pooling for embedding generation. Defaualt = 'mean'.")
    parser.add_argument("--ignore_unknown", default=False, action="store_true", help="Ignore unknown tokens during calculations.")
    parser.add_argument("--n_neighbors", default=5, type=int, help="Number of neighbors for KNN classification.")

    args = parser.parse_args()

    # Ensure max_seq_length is greater than or equal to attention_window
    args.max_seq_length = max(args.max_seq_length, args.attention_window)
    # Round down max_seq_length to the nearest multiple of attention_window
    args.max_seq_length = (args.max_seq_length // args.attention_window) * args.attention_window

    return args

def query_model(rank, model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, genome_labels, cluster_assignments, DDP_active, encoder_only, query_prompt_list, query_genome_labels):
    if DDP_active:
        setup(rank, world_size, args.port)
        #prompt_list = prompt_list[rank]
        sampler = DistributedSampler(prompt_list, num_replicas=world_size, rank=rank, shuffle=False)
        num_workers = 0
        pin_memory = False
        shuffle = False
    else:
        sampler = None
        pin_memory = True
        shuffle = False
        num_workers=1
    
    # load model
    model = LEDForConditionalGeneration(BARTlongformer_config)
    device = rank
    model = model.to(device)
    if DDP_active:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    map_location = None
    if DDP_active:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        dist.barrier()
    
    if map_location != None:
        checkpoint = torch.load(model_path, map_location=map_location)
    else:
        checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # query model with known prompts
    dataset = GenomeDataset(prompt_list, tokenizer, args.max_seq_length, 0, args.global_contig_breaks, False, genome_labels, args.ignore_unknown)
    dataset.attention_window = args.attention_window
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
    
    prompt_list_df = calculate_embedding(model, tokenizer, loader, device, args.max_seq_length, encoder_only, args.outpref, args.pooling)

    # query model with unknown prompts
    dataset = GenomeDataset(query_prompt_list, tokenizer, args.max_seq_length, 0, args.global_contig_breaks, False, query_genome_labels, args.ignore_unknown)
    dataset.attention_window = args.attention_window
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
    
    query_list_df = calculate_embedding(model, tokenizer, loader, device, args.max_seq_length, encoder_only, args.outpref, args.pooling)

    # add known labels and use k-NN to generate labels for unknown
    X_train = prompt_list_df.iloc[:, 1:].values  # Features (N-dimensional embeddings)
    y_train = np.array(query_genome_labels)   # Labels

    # Train the classifier
    knn = KNeighborsClassifier(n_neighbors=args.n_neighbors)
    knn.fit(X_train, y_train)

    # predict from classifier
    X_test = query_list_df.iloc[:, 1:].values
    y_pred = knn.predict(X_test)

    query_list_df['predicted_label'] = y_pred

    if args.query_labels != None:
        # parse real data labels
        query_cluster_assignments = []
        with open(args.query_labels, "r") as i:
            for line in i:
                split_line = line.split(",")
                query_cluster_assignments.append(split_line[1])
        y_test = np.array(query_cluster_assignments)
        unique_labels = np.unique(y_test)
        # Per-class accuracy
        per_label_accuracy = []
        for label in unique_labels:
            # Select only test examples of this class
            idx = y_test == label
            label_acc = accuracy_score(y_test[idx], y_pred[idx])
            per_label_accuracy.append({
                'Label': label,
                'Accuracy': label_acc
            })
        
        # Overall accuracy
        all_acc = accuracy_score(y_test, y_pred)
        per_label_accuracy.append({
                'Label': "overall",
                'Accuracy': all_acc
            })

        # Convert to DataFrame
        per_label_df = pd.DataFrame(per_label_accuracy)

        # Save to TSV
        per_label_df.to_csv(args.outpref + "_per_label_accuracy.tsv", sep='\t', index=False)

def main():
    args = parse_args_script()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    args.max_seq_length = max(args.max_seq_length, args.attention_window)
    # Round down max_seq_length to the nearest multiple of attention_window
    args.max_seq_length = (args.max_seq_length // args.attention_window) * args.attention_window
    device = args.device

    DDP_active = args.DDP

    BARTlongformer_config = LEDConfig(
        vocab_size=vocab_size,
        d_model=args.embed_dim,
        encoder_layers=args.num_layers,
        decoder_layers=args.num_layers,
        encoder_attention_heads=args.num_heads,
        decoder_attention_heads=args.num_heads,
        decoder_ffn_dim=4 * args.embed_dim,
        encoder_ffn_dim=4 * args.embed_dim,
        max_encoder_position_embeddings=args.max_seq_length,
        max_decoder_position_embeddings=args.max_seq_length,
        dropout=args.model_dropout_rate,
        attention_window = args.attention_window
        )
    
    world_size = torch.cuda.device_count()
    if DDP_active:
        if world_size > 0:
            # Use DDP but just one GPU
            if device != None:
                device = torch.device("cuda:{}".format(device))
                world_size = 1
            else:
                device = torch.device("cuda") # Run on a GPU if one is available
            print("{} GPU(s) available, using cuda".format(world_size))
        else:
            print("GPU not available, using cpu.")
            device = torch.device("cpu")
    else:
        if world_size > 0 and device != "cpu":
            device = torch.device("cuda:{}".format(device))
        else:
            device = torch.device("cpu")

    genome_labels = []
    cluster_assignments = []
    with open(args.prompt_labels, "r") as i:
        for line in i:
            split_line = line.split(",")
            genome_name = split_line[0]
            genome_labels.append(genome_name)
            cluster_assignments.append(split_line[1])

    # parse prompt file and additional query files for assignment to clusters
    prompt_list, genome_labels = read_prompt_file(args.prompt_file, genome_labels)
    query_prompt_list, query_genome_labels = read_prompt_file(args.prompt_file, [])

    # randomise
    if args.randomise:
        prompt_list = [genome.split() for genome in prompt_list]
        for genome in prompt_list:
            random.shuffle(genome)
        prompt_list = [" ".join(genome) for genome in prompt_list]

    # remove sequences that are too long or short
    if args.max_input_len != None:
        # for prompt_list
        list_index = [i for i, genome in enumerate(prompt_list) if len(genome.split()) <= args.max_input_len]
        prompt_list = [prompt_list[i] for i in list_index]
        genome_labels = [genome_labels[i] for i in list_index]
        cluster_assignments = [cluster_assignments[i] for i in list_index]

        # for query_list
        list_index = [i for i, genome in enumerate(query_prompt_list) if len(genome.split()) <= args.max_input_len]
        query_prompt_list = [query_prompt_list[i] for i in list_index]
        query_genome_labels = [query_genome_labels[i] for i in list_index]

    if args.min_input_len != None:
        # for prompt_list
        list_index = [i for i, genome in enumerate(prompt_list) if len(genome.split()) <= args.max_input_len]
        prompt_list = [prompt_list[i] for i in list_index]
        genome_labels = [genome_labels[i] for i in list_index]
        cluster_assignments = [cluster_assignments[i] for i in list_index]

        # for query_list
        list_index = [i for i, genome in enumerate(query_prompt_list) if len(genome.split()) <= args.max_input_len]
        query_prompt_list = [query_prompt_list[i] for i in list_index]
        query_genome_labels = [query_genome_labels[i] for i in list_index]

    return_list = []
    if DDP_active:
        #prompt_list = split_prompts(prompt_list, world_size)
        with Manager() as manager:
            mp_list = manager.list()
            mp.spawn(query_model,
                    args=(args.model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, genome_labels, cluster_assignments, DDP_active, args.encoder_only, query_prompt_list, query_genome_labels),
                    nprocs=world_size,
                    join=True)
            return_list = list(mp_list)
    else:
        query_model(device, args.model_path, 1, args, BARTlongformer_config, tokenizer, prompt_list, genome_labels, cluster_assignments, DDP_active, args.encoder_only, query_prompt_list, query_genome_labels)

    if DDP_active:
        cleanup()

if __name__ == "__main__":
    main()