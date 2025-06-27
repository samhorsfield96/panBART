import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import LEDConfig, LEDForConditionalGeneration, LEDTokenizer, logging
import numpy as np
import argparse
from tqdm import tqdm
import re
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from multiprocessing import Manager
import torch.multiprocessing as mp
from panGPT import setup, cleanup, GenomeDataset, load_dataset
import random
from panGPT import mask_integers, flip_contig
from torch.utils.data import DataLoader, DistributedSampler

logging.set_verbosity_error()

def jaccard_distance(A, B):
    set_A = set(abs(g) for g in A)
    set_B = set(abs(g) for g in B)
    intersection = len(set_A & set_B)
    union = len(set_A | set_B)
    return 1 - intersection / union if union > 0 else 0

def breakpoint_distance(A, B):
    # Map gene to position in A
    gene_to_pos_A = {abs(g): i for i, g in enumerate(A)}
    # Keep only common genes
    common_genes = set(abs(g) for g in A) & set(abs(g) for g in B)
    A_filtered = [g for g in A if abs(g) in common_genes]
    B_filtered = [g for g in B if abs(g) in common_genes]

    # Map B genes to A positions
    pos_B = [gene_to_pos_A[abs(g)] for g in B_filtered]
    # Add sentinels at start and end
    pos_B = [-1] + pos_B + [len(A_filtered)]
    
    breakpoints = sum(1 for i in range(len(pos_B)-1) if pos_B[i+1] - pos_B[i] != 1)
    max_possible = len(A_filtered) + 1  # Normalizing factor
    return breakpoints / max_possible if max_possible > 0 else 0

def shuffle_genome(genome):
    split_genome = genome.strip().split("_")
    # randomise contig order and flip randomly

    #flip_contigs = [random.random() < 0.5 for _ in range(len(split_genome))]
    #print(split_genome)
    #print(flip_contigs)

    #for index, contig in enumerate(split_genome):
    #    if flip_contigs[index]:
    #        split_genome[index] = flip_contig(contig.strip())
    #    else:
    #        split_genome[index] = contig.strip()

    #print(split_genome)
    random.shuffle(split_genome)
    return split_genome

def parse_args():
    """
    Parse command-line arguments.

    This function parses the command-line arguments provided by the user and returns
    a Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Token prediction with a Transformer or Reformer model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer file.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the text file containing the prompt.")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for prediction. if unset, will take most likely value.")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers.")
    parser.add_argument("--max_seq_length", type=int, default=16384, help="Maximum sequence length.")
    parser.add_argument("--model_dropout_rate", type=float, default=0.2, help="Dropout rate for the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Maximum batch size for simulation. Default = 16")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--attention_window", type=int, default=512, help="Attention window size in the Longformer model (default: 512)")
    parser.add_argument("--prop_masked", type=float, default=0.15, help="Proportion of prompt to be masked for encoder. Default = 0.15")
    parser.add_argument("--prop_prompt_kept", type=float, default=1.0, help="Proportion of prompt from start to be kept before encoding. If 0.0, keeps one contig break marker. Default = 0.0")
    parser.add_argument("--shuffle_genomes", default=False, action="store_true", help="Shuffle order of contigs for prompt.")
    parser.add_argument("--max_input_len", type=int, default=None, help="Maximum length of input sequence. No limit if not set.")
    parser.add_argument("--min_input_len", type=int, default=None, help="Minimum length of input sequence. No limit if not set.")
    parser.add_argument("--num_seq", type=int, default=1, help="Number of simulations per prompt sequence. Default = 1")
    parser.add_argument("--outpref", type=str, default="simulated_genomes", help="Output file for simulated genomes. Default = 'simulated_genomes'")
    parser.add_argument("--DDP", action="store_true", default=False, help="Multiple GPUs used via DDP during training.")
    parser.add_argument("--encoder_only", default=False, action="store_true", help="Prompt using encoder input only.")
    parser.add_argument("--generate", default=False, action="store_true", help="Generate iteratively instead of as a block.")
    parser.add_argument("--global_contig_breaks", default=False, action="store_true", help="Attend globally to contig breaks. Default is local only.")
    parser.add_argument("--port", default="12356", type=str, help="GPU port for DDP. Default=12356")

    args = parser.parse_args()

    # Ensure max_seq_length is greater than or equal to attention_window
    args.max_seq_length = max(args.max_seq_length, args.attention_window)
    # Round down max_seq_length to the nearest multiple of attention_window
    args.max_seq_length = (args.max_seq_length // args.attention_window) * args.attention_window

    return args

def load_model(embed_dim, num_heads, num_layers, max_seq_length, device, vocab_size, attention_window, model_dropout_rate):

    BARTlongformer_config = LEDConfig(
        vocab_size=vocab_size,
        d_model=embed_dim,
        encoder_layers=num_layers,
        decoder_layers=num_layers,
        encoder_attention_heads=num_heads,
        decoder_attention_heads=num_heads,
        decoder_ffn_dim=4 * embed_dim,
        encoder_ffn_dim=4 * embed_dim,
        max_encoder_position_embeddings=max_seq_length,
        max_decoder_position_embeddings=max_seq_length,
        dropout=model_dropout_rate,
        attention_window = attention_window
        )
    model = LEDForConditionalGeneration(BARTlongformer_config)
    model.config.use_cache = True
    return model

def predict_next_tokens_BART(model, tokenizer, loader, device, temperature, num_seq, max_seq_length, encoder_only, generate, DDP_active, prompt_list_truth):
    model.eval()

    predictions = []
    jaccard_accuracy_list = []
    breakpoint_accuracy_list = []
    with torch.no_grad():
        # repeat for number of sequences required. Means each sequences is masked in different ways
        for _ in range(num_seq):
            for decoder_input, encoder_input, labels, decoder_attention_mask, encoder_attention_mask, global_attention_mask, idx in loader:  # Correctly unpack the tuples returned by the DataLoader
                
                # get real sequence to compare to
                real_seq = prompt_list_truth[idx]

                total_len = len(real_seq) #encoder_input.size(1)
                current_sequence = []
                current_accuracy_list = []
                for i in range(0, total_len, max_seq_length):
                    # print("max_seq_length exceeded: {}".format(total_len > max_seq_length))

                    if not generate:
                        if encoder_only:
                            batch_encoder_input, batch_encoder_attention_mask, batch_global_attention_mask = encoder_input[:, i:i + max_seq_length].to(device), encoder_attention_mask[:, i:i + max_seq_length].to(device), global_attention_mask[:, i:i + max_seq_length].to(device) # Move data to the appropriate device
                            
                            outputs = model(input_ids=batch_encoder_input, attention_mask=batch_encoder_attention_mask, global_attention_mask=batch_global_attention_mask).logits  # Generate predictions
                        
                            # Free GPU memory
                            del batch_encoder_input
                            del batch_encoder_attention_mask
                            del batch_global_attention_mask

                        else:
                            batch_decoder_input, batch_encoder_input, batch_decoder_attention_mask, batch_encoder_attention_mask, batch_global_attention_mask = decoder_input[:, i:i + max_seq_length].to(device), encoder_input[:, i:i + max_seq_length].to(device), decoder_attention_mask[:, i:i + max_seq_length].to(device), encoder_attention_mask[:, i:i + max_seq_length].to(device), global_attention_mask[:, i:i + max_seq_length].to(device) # Move data to the appropriate device
                            
                            outputs = model(input_ids=batch_encoder_input, attention_mask=batch_encoder_attention_mask, decoder_input_ids=batch_decoder_input, decoder_attention_mask=batch_decoder_attention_mask, global_attention_mask=batch_global_attention_mask).logits  # Generate predictions
                            
                            # Free GPU memory
                            del batch_encoder_input
                            del batch_encoder_attention_mask
                            del batch_decoder_input
                            del batch_decoder_attention_mask
                            del batch_global_attention_mask

                        if temperature != None:
                            # generate predictions with Temperature
                            scaled_logits = outputs / temperature
                            probabilities = F.softmax(scaled_logits, dim=2)
                            preds = torch.multinomial(probabilities.view(-1, probabilities.size(-1)), 1)
                            # Reshape the sampled token IDs to match the batch size and sequence length
                            preds = preds.view(outputs.size(0), outputs.size(1))
                        else:
                        # take highest value
                            preds = outputs.argmax(dim=-1)  # Get predicted classes
                        
                        batch_labels = labels[:, i:i + max_seq_length].to(device)

                        # ignore padding positions
                        mask = batch_labels != -100
                        preds = preds[mask]
                        batch_labels = batch_labels[mask]

                        # calculate accuracy, ignore
                        correct = (preds == batch_labels).sum().item()
                        accuracy = correct / batch_labels.numel()

                        #print("matches:")
                        #print((preds == batch_labels).type(torch.uint8).tolist())

                    else:
                        #generated iteratively (very slow)
                        batch_encoder_input, batch_encoder_attention_mask, batch_global_attention_mask, batch_decoder_input = encoder_input[:, i:i + max_seq_length].to(device), encoder_attention_mask[:, i:i + max_seq_length].to(device), global_attention_mask[:, i:i + max_seq_length].to(device), decoder_input[:, i:i + max_seq_length].to(device) # Move data to the appropriate device
                        if DDP_active:
                            preds = model.module.generate(
                                batch_encoder_input,
                                global_attention_mask=batch_global_attention_mask,
                                attention_mask=batch_encoder_attention_mask,
                                # is max length here correct?
                                max_length=batch_encoder_input.shape[1],
                                temperature=temperature,
                                num_return_sequences=1,   # Number of sequences to return
                                do_sample=True,
                                decoder_start_token_id=batch_decoder_input[:, 0]
                            )
                        else:
                            preds = model.generate(
                                batch_encoder_input,
                                global_attention_mask=batch_global_attention_mask,
                                attention_mask=batch_encoder_attention_mask,
                                # is max length here correct?
                                max_length=batch_encoder_input.shape[1],
                                temperature=temperature,
                                num_return_sequences=1,   # Number of sequences to return
                                do_sample=True,
                                decoder_start_token_id=batch_decoder_input[:, 0]
                            )

                        batch_labels = labels[:, i:i + max_seq_length].to(device)

                        # ignore padding positions
                        mask = batch_labels[:, :-1] != -100
                        preds = preds[:, 1:][mask]
                        batch_labels = batch_labels[:, :-1][mask]

                        # calculate accuracy, ignore
                        correct = (preds == batch_labels).sum().item()
                        accuracy = correct / batch_labels.numel()

                        #print("matches:")
                        #print((preds == batch_labels).type(torch.uint8).tolist())

                        # preds = batch_decoder_input[:, 0].tolist()
                        # for j in tqdm(range(batch_encoder_input.shape[1]), desc="Token number", total=batch_encoder_input.shape[1]):
                        #     tokens = torch.tensor([preds]).to(device)
                        #     outputs = model(input_ids=batch_encoder_input, attention_mask=batch_encoder_attention_mask, decoder_input_ids=tokens, global_attention_mask=batch_global_attention_mask).logits
                        #     scaled_logits = outputs[0, j, :] / temperature
                        #     probabilities = F.softmax(scaled_logits, dim=-1)
                        #     next_token_id = torch.multinomial(probabilities, 1).item()
                        #     #print(next_token_id)
                        #     preds.append(next_token_id)
                        # #print(preds)
                        # preds = torch.tensor([preds[1:]]).to(device)

                    #torch.cuda.empty_cache()

                    current_accuracy_list.append(accuracy)
                    #print("accuracy: {}".format(round(accuracy, 4)))
                    #print("loss: {}".format(loss.item()))
                    #print("preds:")
                    #print(preds.tolist())
                    #print("labels:")
                    #print(batch_labels.tolist())

                    current_sequence.append(preds)
            
                # concatenate predictions and get average accuracy
                #current_accuracy = sum(current_accuracy_list) / len(current_accuracy_list)
                #print("accuracy: {}".format(round(current_accuracy, 4)))
                #accuracy_list.append(round(current_accuracy, 4))
                current_sequence = [tensor.unsqueeze(0) if tensor.ndim == 1 else tensor for tensor in current_sequence]
                sequence = torch.cat(current_sequence, dim=1).tolist()[0]
                
                sequence = tokenizer.decode(sequence, skip_special_tokens=True)

                # get jaccard and breakpoint distance
                jaccard_dist = jaccard_distance(sequence, real_seq)
                break_dist = breakpoint_distance(sequence, real_seq)
                jaccard_accuracy_list.append(jaccard_dist)
                breakpoint_accuracy_list.append(break_dist)

                #print(sequence)
                predictions.append(sequence)
   
    return_list = [(jdist, breakdist, seq) for acc, seq in zip(jaccard_accuracy_list, breakpoint_accuracy_list, predictions)]

    return return_list

def query_model(rank, model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, encoder_only, prompt_list_truth, return_list):
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
    
    dataset = GenomeDataset(prompt_list, tokenizer, args.max_seq_length, args.prop_masked, args.global_contig_breaks, False, ID_list=list(range(0,len(prompt_list_truth))))
    dataset.attention_window = args.attention_window
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
    
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

    master_process = rank == 0

    predicted_text = predict_next_tokens_BART(model, tokenizer, loader, device, args.temperature, args.num_seq, args.max_seq_length, encoder_only, args.generate, DDP_active, prompt_list_truth)
    
    return_list.extend(predicted_text)

        
def main():
    args = parse_args()

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

    prompt_list = load_dataset(args.prompt_file)

    # remove sequences that are too long or short
    if args.max_input_len != None:
        # len_list = [len(genome.split()) for genome in prompt_list]
        # print(len_list)
        prompt_list = [genome for genome in prompt_list if len(genome.split()) <= args.max_input_len]

    if args.min_input_len != None:
        # len_list = [len(genome.split()) for genome in prompt_list]
        # print(len_list)
        prompt_list = [genome for genome in prompt_list if len(genome.split()) >= args.min_input_len]

    if args.shuffle_genomes:
        prompt_list = [shuffle_genome(genome) for genone in prompt_list]

    prompt_list_truth = prompt_list
    if args.prop_prompt_removed < 1.0:
        if args.prop_prompt_removed == 0.0:
            prompt_list = ["_" for genone in prompt_list]
        else:
            prompt_list = [genome[0:int(len(genome) * args.prop_prompt_removed)] for genome in prompt_list]

    return_list = []
    if DDP_active:
        with Manager() as manager:
            mp_list = manager.list()
            mp.spawn(query_model,
                    args=(args.model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, args.encoder_only, prompt_list_truth, mp_list),
                    nprocs=world_size,
                    join=True)
            return_list = list(mp_list)
    else:
        query_model(device, args.model_path, 1, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, args.encoder_only, prompt_list_truth, return_list)
    
    with open(args.outpref + "_seq.txt", "w") as f1, open(args.outpref + "_acc.txt", "w") as f2:
        f2.write("Prop_masked\tTemperature\tIndex\Jaccard_distance\tBreakpoint_distance\n")
        for index, (jdist, break_dist, seq) in enumerate(return_list):
            f1.write(">" + str(index) + " jaccard_distance: " + str(jdist) + " breakpoint_distance: " + str(break_dist) + "\n" + str(seq) + "\n")
            f2.write(str(args.prop_masked) + "\t" + str(args.temperature) + "\t" + str(index) + "\t" + str(jdist) + "\t" + str(break_dist) + "\n")

    if DDP_active:
        cleanup()

if __name__ == "__main__":
    main()
