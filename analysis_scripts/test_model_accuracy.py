from panGPT import *

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler()
    ],
)

def test_model(rank, world_size, args, BARTlongformer_config, test_genomes, tokenizer, vocab_size, DDP_active=False):
    if DDP_active:
        setup(rank, world_size, args.port)
    
    # determine number of GPUs to use
    #num_gpus = torch.cuda.device_count()
    model = LEDForConditionalGeneration(BARTlongformer_config)
    if args.gradient_checkpointing == True and DDP_active == False:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    device = rank
    model = model.to(device)
    if DDP_active:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        test_sampler = DistributedSampler(test_genomes, num_replicas=world_size, rank=rank)
        args.num_workers = 0
        pin_memory = False
        shuffle = False
    else:
        test_sampler = None, None, None
        pin_memory = True
        shuffle = True

    map_location = None
    if DDP_active:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        dist.barrier()

    # load model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate * math.sqrt(world_size), weight_decay=args.weight_decay) # How are we trying to optimizer it?
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_scheduler_factor, patience=args.lr_patience) # taking big, then small steps
    start_epoch, is_checkpoint_loaded = load_checkpoint(model, optimizer, lr_scheduler, args.model_save_path, False, map_location)

    # get loss object
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # generate genome datasets
    test_dataset = GenomeDataset(test_genomes, tokenizer, args.max_seq_length, 0, args.global_contig_breaks, ignore_unknown=args.ignore_unknown)
    test_dataset.attention_window = args.attention_window
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, sampler=test_sampler)
    #test_loader.sampler.set_epoch(epoch)
    test_dataset_size = len(test_loader.dataset)  # Store the size of the test dataset
    test_loader = tqdm(test_loader, desc="Testing", unit="batch")
    # Test Model Loop
    total_test_loss, total_test_accuracy, test_precision, test_recall, test_f1 = validate_model(test_loader, model, criterion, device, vocab_size, args.encoder_only)
    
    total_test_loss_tensor = torch.tensor(total_test_loss).to(rank)
    total_accuracy_tensor = torch.tensor(total_test_accuracy).to(rank)
    test_precision_tensor = torch.tensor(test_precision).to(rank)
    test_recall_tensor = torch.tensor(test_recall).to(rank)
    test_f1_tensor = torch.tensor(test_f1).to(rank)

    # get results from all GPUs
    if DDP_active:
        dist.all_reduce(total_test_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_accuracy_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_precision_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_recall_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_f1_tensor, op=dist.ReduceOp.SUM)
    
    avg_test_loss = total_test_loss_tensor.item() / test_dataset_size
    test_perplexity = torch.exp(torch.tensor(avg_test_loss))
    test_accuracy = total_accuracy_tensor.item() / test_dataset_size
    test_precision = test_precision_tensor.item() / world_size
    test_recall = test_recall_tensor.item() / world_size
    test_f1 = test_f1_tensor.item() / world_size

    # Log test metrics
    if (DDP_active and rank == 0) or DDP_active == False:  # Only rank 0 should write logs
        logging.info(f'Test Loss: {avg_test_loss}, Perplexity: {test_perplexity}, Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}')

    if DDP_active:
        cleanup()

def main():
    args = parse_args()

    params = vars(args)  # Convert the parsed arguments to a dictionary

    test_file = args.test_file
    device = args.device
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    num_layers = args.num_layers
    max_vocab_size = args.max_vocab_size
    model_save_path = args.model_save_path
    tokenizer_path = args.tokenizer_path
    seed = args.seed
    max_input_len = args.max_input_len
    min_input_len = args.min_input_len
    model_dropout_rate = args.model_dropout_rate

    # Check if max_seq_length is a multiple of attention_window when using Longformer
    #if (model_type == "longformer" or model_type == "BARTlongformer") and max_seq_length % attention_window != 0:
    if args.max_seq_length % args.attention_window != 0:
        logging.info(f"Error: When using the LED model, the maximum sequence length (args.max_seq_length) must be a multiple of the attention window size (attention_window).")
        logging.info(f"Current values: max_seq_length = {args.max_seq_length}, attention_window = {args.attention_window}")
        logging.info("Please adjust these values and try again.")
        exit(1)

    # read in pre-split genomes
    test_genomes = load_dataset(test_file)
    if max_input_len != None:
        test_genomes = [genome for genome in test_genomes if len(genome.split()) <= max_input_len]

    if min_input_len != None:
        test_genomes = [genome for genome in test_genomes if len(genome.split()) >= min_input_len]

    # combine for tokenizer
    genomes = test_genomes

    # genenerate reversed contigs for tokenizer
    reversed_genomes = []
    for genome in genomes:
        split_genome = genome.strip().split("_")
        # randomise contig order and flip randomly
        for index, contig in enumerate(split_genome):
            split_genome[index] = flip_contig(contig.strip())
        
        rev_genome = " _ ".join(split_genome)
        reversed_genomes.append(rev_genome)

    #print(reversed_genomes)

    if model_save_path and not os.path.isdir(os.path.dirname(model_save_path)):
        print(f"Error: The directory for model save path '{model_save_path}' does not exist.")
        exit(1)
    
    # generate tokenizer information
    set_seed(args.seed)
    print_parameters_table(params)
    
    sequence_lengths = [len(genome.split()) for genome in genomes]
    num_sequences = len(genomes)
    min_sequence_length = min(sequence_lengths)
    max_sequence_length = max(sequence_lengths)
    avg_sequence_length = sum(sequence_lengths) / num_sequences

    logging.info(
        f"Dataset loaded: {num_sequences} sequences\n"
        f"Sequence lengths - Min: {min_sequence_length}, Max: {max_sequence_length}, Avg: {avg_sequence_length:.2f}"
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    BARTlongformer_config = LEDConfig(
        vocab_size=vocab_size,
        d_model=embed_dim,
        encoder_layers=num_layers,
        decoder_layers=num_layers,
        encoder_attention_heads=num_heads,
        decoder_attention_heads=num_heads,
        decoder_ffn_dim=4 * embed_dim,
        encoder_ffn_dim=4 * embed_dim,
        max_encoder_position_embeddings=args.max_seq_length,
        max_decoder_position_embeddings=args.max_seq_length,
        dropout=model_dropout_rate,
        attention_window = args.attention_window,
        pad_token_id=tokenizer.encode("<pad>").ids[0],
        bos_token_id=tokenizer.encode("<s>").ids[0],
        eos_token_id=tokenizer.encode("</s>").ids[0]
        )

    DDP_active = False
    world_size = torch.cuda.device_count()
    if device is None:
        if world_size > 0:
            print("{} GPU(s) available, using cuda".format(world_size))

            device = torch.device("cuda") # Run on a GPU if one is available
            DDP_active = True
        else:
            print("GPU not available, using cpu.")
            device = torch.device("cpu")
    else:
        if world_size > 0 and device != "cpu":
            device = torch.device("cuda:{}".format(device))
        else:
            device = torch.device("cpu")

    # delete genomes list from memory
    del genomes

    print(f"vocab_size: {vocab_size} | embed_dim: {embed_dim} | num_heads: {num_heads} | num_layers: {num_layers} | max_seq_length: {args.max_seq_length}", flush=True)
    if DDP_active:
        mp.spawn(test_model,
                args=(world_size, args, BARTlongformer_config, test_genomes, tokenizer, vocab_size, DDP_active),
                nprocs=world_size,
                join=True)
    else:
        test_model(device, 1, args, BARTlongformer_config, test_genomes, tokenizer, vocab_size, DDP_active)

if __name__ == "__main__":
    main()