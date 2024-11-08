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
from panGPT import setup, cleanup
import random
from torch.utils.data import DataLoader, DistributedSampler
from panPrompt import GenomeDataset, load_dataset
from collections import defaultdict
from torch.nn.functional import cosine_similarity
import pandas as pd

logging.set_verbosity_error()

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
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers.")
    parser.add_argument("--max_seq_length", type=int, default=16384, help="Maximum sequence length.")
    parser.add_argument("--model_dropout_rate", type=float, default=0.2, help="Dropout rate for the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Maximum batch size for simulation. Default = 16")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--attention_window", type=int, default=512, help="Attention window size in the Longformer model (default: 512)")
    parser.add_argument("--max_input_len", type=int, default=None, help="Maximum length of input sequence. No limit if not set.")
    parser.add_argument("--min_input_len", type=int, default=None, help="Minimum length of input sequence. No limit if not set.")
    parser.add_argument("--outpref", type=str, default="simulated_genomes", help="Output prefix for simulated genomes. Default = 'simulated_genomes'")
    parser.add_argument("--DDP", action="store_true", default=False, help="Multiple GPUs used via DDP during training.")
    parser.add_argument("--encoder_only", default=False, action="store_true", help="Prompt using encoder input only.")
    parser.add_argument("--randomise", default=False, action="store_true", help="Randomise sequence for upon input.")

    args = parser.parse_args()

    # Ensure max_seq_length is greater than or equal to attention_window
    args.max_seq_length = max(args.max_seq_length, args.attention_window)
    # Round down max_seq_length to the nearest multiple of attention_window
    args.max_seq_length = (args.max_seq_length // args.attention_window) * args.attention_window

    return args

def pad_input(input, max_length, pad_token_id, labels=False):

    len_masked = len(input)
    # only pad if necessary
    if len_masked >= max_length:
        pass
    else:
        if labels == False:
            input.extend([pad_token_id] * (max_length - len_masked))
        else:
            input.extend([-100] * (max_length - len_masked))

    return input

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

def get_pseudolikelihood(outputs, i, batch_encoder_input):

    logits = outputs[0, i]

    # Get the probability of the original token
    original_token_id = batch_encoder_input[0, i]
    token_prob = torch.softmax(logits, dim=-1)[original_token_id].item()

    # Add the log probability to the total log-pseudo-likelihood
    log_pseudo_likelihood = torch.log(torch.tensor(token_prob)).to("cpu").item()
    
    # move elements to cpu
    original_token_id = original_token_id.to("cpu").item()
    
    return log_pseudo_likelihood, original_token_id

def calculate_cosine(model, tokenizer, loader, device, max_seq_length, encoder_only, outpref):
    model.eval()
    mask_token = tokenizer.encode("<mask>").ids[0]
    pad_token = tokenizer.encode("<pad>").ids[0]

    # Get the vocabulary dictionary: {token: token_id}
    vocab_dict = tokenizer.get_vocab()

    # Sort tokens by token ID to get them in the order they appear in the logits
    sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])

    # Extract the tokens in order of their IDs
    labels = [token for token, _ in sorted_vocab]
    #print(len(labels))

    hidden_size = model.config.hidden_size
    token_embeddings_sum = defaultdict(lambda: torch.zeros(hidden_size))
    token_counts = defaultdict(int)

    #label_set = set()

    with torch.no_grad():
        # repeat for number of sequences required. Means each sequences is masked in different ways
        for decoder_input, encoder_input, labels, decoder_attention_mask, encoder_attention_mask, global_attention_mask in loader:  # Correctly unpack the tuples returned by the DataLoader

            total_len = encoder_input.size(1)
            log_pseudo_likelihood = 0.0
            for i in range(0, total_len, max_seq_length):
                # print("max_seq_length exceeded: {}".format(total_len > max_seq_length))

                #print(labels)
                if encoder_only:
                    batch_encoder_input, batch_encoder_attention_mask, batch_global_attention_mask = encoder_input[:, i:i + max_seq_length].to(device), encoder_attention_mask[:, i:i + max_seq_length].to(device), global_attention_mask[:, i:i + max_seq_length].to(device) # Move data to the appropriate device
                    
                    outputs = model(input_ids=batch_encoder_input, attention_mask=batch_encoder_attention_mask, global_attention_mask=batch_global_attention_mask, output_hidden_states=True)  # Generate predictions

                    # Free GPU memory
                    del batch_encoder_input
                    del batch_encoder_attention_mask
                    del batch_global_attention_mask

                else:
                    batch_decoder_input, batch_encoder_input, batch_decoder_attention_mask, batch_encoder_attention_mask, batch_global_attention_mask = decoder_input[:, i:i + max_seq_length].to(device), encoder_input[:, i:i + max_seq_length].to(device), decoder_attention_mask[:, i:i + max_seq_length].to(device), encoder_attention_mask[:, i:i + max_seq_length].to(device), global_attention_mask[:, i:i + max_seq_length].to(device) # Move data to the appropriate device
                    
                    
                    outputs = model(input_ids=batch_encoder_input, attention_mask=batch_encoder_attention_mask, decoder_input_ids=batch_decoder_input, decoder_attention_mask=batch_decoder_attention_mask, global_attention_mask=batch_global_attention_mask, output_hidden_states=True)  # Generate predictions
                    #decoder_hidden_states = outputs.decoder_hidden_states
                    encoder_last_hidden_state = outputs.encoder_last_hidden_state.detach().cpu().squeeze(0)

                    # Aggregate embeddings and update counts for each token
                    for idx, token_id in enumerate(batch_encoder_input.squeeze(0)):
                        _token_id = token_id.item()
                        token_embeddings_sum[_token_id] += encoder_last_hidden_state[idx]
                        token_counts[_token_id] += 1
                        #label_set.add(_token_id)

                    # Free GPU memory
                    del batch_encoder_input
                    del batch_encoder_attention_mask
                    del batch_decoder_input
                    del batch_decoder_attention_mask
                    del batch_global_attention_mask

    #print(len(label_set))
    #print(label_set)
    # calculate average embeddings
    token_avg_embeddings = {}
    for token_id, embedding_sum in token_embeddings_sum.items():
        count = token_counts[token_id]
        if count > 0:
            token_avg_embeddings[token_id] = embedding_sum / count

    # generate torch tensor of average embeddings
    token_list = list(token_avg_embeddings.values())
    tokens = [tokenizer.decode([x]) for x in token_avg_embeddings.keys()]
    token_embeddings = torch.stack(token_list)  # Shape: (num_tokens, hidden_size)
    # print("token_embeddings")
    # print(token_embeddings)
    # print(token_embeddings.shape)

    # Normalize the embeddings
    normalized_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=1)
    # print("normalized_embeddings")
    # print(normalized_embeddings)
    # print(normalized_embeddings.shape)


    # Compute pairwise cosine similarities
    cosine_similarities = torch.mm(normalized_embeddings, normalized_embeddings.T)
    # print("cosine_similarities")
    # print(cosine_similarities)
    # print(cosine_similarities.shape)

    # Step 3: Construct the pairwise cosine similarity matrix
    cosine_similarities = torch.clamp(cosine_similarities, min=-1.0, max=1.0)
    cosine_similarity_matrix = cosine_similarities.squeeze(0).cpu().numpy()

    # print("cosine_similarity_matrix")
    # print(cosine_similarity_matrix)
    # print(cosine_similarities.shape)
    
    df = pd.DataFrame(cosine_similarity_matrix, index=tokens, columns=tokens)

    # Display the DataFrame
    df.to_csv(outpref + ".csv", index=True)


def read_prompt_file(file_path):
    prompt_list = []
    with open(file_path, 'r') as file:
        for line in file:
            prompt_list.append(line.strip())
    return prompt_list

def split_prompts(prompts, world_size):
    # Split prompts into approximately equal chunks for each GPU
    chunk_size = len(prompts) // world_size
    return [prompts[i * chunk_size:(i + 1) * chunk_size] for i in range(world_size)]

def query_model(rank, model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, encoder_only):
    if DDP_active:
        setup(rank, world_size)
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
    
    dataset = GenomeDataset(prompt_list, tokenizer, args.max_seq_length, 0)
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

    calculate_cosine(model, tokenizer, loader, device, args.max_seq_length, encoder_only, args.outpref)

        
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

    # randomise
    if args.randomise:
        prompt_list = [genome.split() for genome in prompt_list]
        for genome in prompt_list:
            random.shuffle(genome)
        prompt_list = [" ".join(genome) for genome in prompt_list]

    # remove sequences that are too long or short
    if args.max_input_len != None:
        # len_list = [len(genome.split()) for genome in prompt_list]
        # print(len_list)
        prompt_list = [genome for genome in prompt_list if len(genome.split()) <= args.max_input_len]

    if args.min_input_len != None:
        # len_list = [len(genome.split()) for genome in prompt_list]
        # print(len_list)
        prompt_list = [genome for genome in prompt_list if len(genome.split()) >= args.min_input_len]

    return_list = []
    if DDP_active:
        #prompt_list = split_prompts(prompt_list, world_size)
        with Manager() as manager:
            mp_list = manager.list()
            mp.spawn(query_model,
                    args=(args.model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, args.encoder_only),
                    nprocs=world_size,
                    join=True)
            return_list = list(mp_list)
    else:
        query_model(device, args.model_path, 1, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, args.encoder_only)

    if DDP_active:
        cleanup()

if __name__ == "__main__":
    main()
