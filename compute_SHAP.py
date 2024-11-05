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
from functools import partial
import shap
import scipy as sp
from panPrompt import mask_integers
import pandas as pd

logging.set_verbosity_error()

def pad_input(input, max_seq_length, pad_token_id, labels=False):

    len_masked = len(input)
    # only pad if necessary
    if len_masked >= max_seq_length:
        pass
    else:
        if labels == False:
            input.extend([pad_token_id] * (max_seq_length - len_masked))
        else:
            input.extend([-100] * (max_seq_length - len_masked))

    return input

def tokenise_input(text, tokenizer, max_seq_length, pad_token, mask_token):
    input = tokenizer.encode(text).ids

    # mask original text
    text = tokenizer.decode(input[1:], skip_special_tokens=False)
    text_masked = mask_integers(text, 0)

    encoder_input = tokenizer.encode(text_masked).ids
    decoder_input = input[:-1]

    len_decoder = len(decoder_input)
    decoder_input = pad_input(decoder_input, max_seq_length, pad_token)

    decoder_attention_mask = torch.ones(len(decoder_input), dtype=torch.long)
    decoder_attention_mask[len_decoder:] = 0

    # merge consecutive masks into single mask token
    # might be issue, might not need to block mask
    encoder_input = ' '.join([str(i) for i in encoder_input])
    pattern = f'({mask_token} )+'
    encoder_input = re.sub(pattern, str(mask_token) + ' ', encoder_input)
    pattern = f'( {mask_token})+'
    encoder_input = re.sub(pattern, ' ' + str(mask_token), encoder_input)
    encoder_input = [int(i) for i in encoder_input.split()]

    len_masked = len(encoder_input)
    encoder_input = pad_input(encoder_input, max_seq_length, pad_token)

    encoder_attention_mask = torch.ones(len(encoder_input), dtype=torch.long)
    encoder_attention_mask[len_masked:] = 0

    # attend to all contig breaks
    global_attention_mask = torch.zeros(len(encoder_input), dtype=torch.long)
    break_idx = np.flatnonzero(np.array(encoder_input) == int(tokenizer.encode("_").ids[0]))
    global_attention_mask[break_idx] = 1

    return torch.tensor(decoder_input, dtype=torch.long).unsqueeze(0), torch.tensor(encoder_input, dtype=torch.long).unsqueeze(0), decoder_attention_mask.unsqueeze(0), encoder_attention_mask.unsqueeze(0), global_attention_mask.unsqueeze(0)

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
    parser.add_argument("--target-token", type=str, required=True, help="Target token to search for")
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
    parser.add_argument("--seed", default=42, type=int, help="Seed for randomisation")

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


# this defines an explicit python function that takes a list of strings and outputs scores for each class
def f(x, model, device, tokenizer, max_seq_length, pad_token, mask_token, pos, encoder_only=False):
    outputs = []
    model.eval()
    for _x in x:
        #print(_x)
        encoder_input, decoder_input, decoder_attention_mask, encoder_attention_mask, global_attention_mask = tokenise_input(_x, tokenizer, max_seq_length, pad_token, mask_token)
        #print(encoder_input)
        #print(pos)
        if encoder_only:
            batch_encoder_input, batch_encoder_attention_mask, batch_global_attention_mask = encoder_input[:, 0:max_seq_length].to(device), encoder_attention_mask[:, 0:max_seq_length].to(device), global_attention_mask[:, 0:max_seq_length].to(device)
            output = model(input_ids=batch_encoder_input, attention_mask=batch_encoder_attention_mask, global_attention_mask=batch_global_attention_mask).logits.detach().cpu().numpy()
        else:
            batch_decoder_input, batch_encoder_input, batch_decoder_attention_mask, batch_encoder_attention_mask, batch_global_attention_mask = decoder_input[:, 0:max_seq_length].to(device), encoder_input[:, 0:max_seq_length].to(device), decoder_attention_mask[:, 0:max_seq_length].to(device), encoder_attention_mask[:, 0:max_seq_length].to(device), global_attention_mask[:, 0:max_seq_length].to(device)
            output = model(input_ids=batch_encoder_input, attention_mask=batch_encoder_attention_mask, decoder_input_ids=batch_decoder_input, decoder_attention_mask=batch_decoder_attention_mask, global_attention_mask=batch_global_attention_mask).logits.detach().cpu().numpy()
        
        #print(output)
        outputs.append(output[0][pos])

    # save all scores in same output
    #outputs = output[0]
    outputs = np.array(outputs)
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores)
    #print(val)
    return val

#may need to change this to use the correct tokenizer
def custom_tokenizer(s, tokenizer, return_offsets_mapping=True):
    """Wraps a Tokenizers tokenizer to conform to SHAP's expectations."""
    
    # Tokenize using the Tokenizers package
    encoding = tokenizer.encode(s)
    
    # Extract input_ids and offset mapping
    input_ids = encoding.ids
    if return_offsets_mapping:
        offset_ranges = encoding.offsets
    else:
        offset_ranges = None
    
    # Format the output to match the transformers-like format
    out = {"input_ids": input_ids}
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    #print(s)
    #print(out["offset_mapping"])
    return out

def calculate_SHAP(model, tokenizer, prompt_list, device, max_seq_length, encoder_only, target_token, outpref, seed):
    # follow this example https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/sentiment_analysis/Using%20custom%20functions%20and%20tokenizers.html
    mask_token = tokenizer.encode("<mask>").ids[0]
    pad_token = tokenizer.encode("<pad>").ids[0]
    target_token_encoded = tokenizer.encode(target_token).ids[0]

    # Get the vocabulary dictionary: {token: token_id}
    vocab_dict = tokenizer.get_vocab()

    # Sort tokens by token ID to get them in the order they appear in the logits
    sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])

    # Extract the tokens in order of their IDs
    labels = [token for token, _ in sorted_vocab]
    #print(labels)

    # create partial tokenizer
    tokenizer_partial = partial(custom_tokenizer, tokenizer=tokenizer)
    # issue might be how masker is working, if it truncates sequences rather than just masking positions.
    masker = shap.maskers.Text(tokenizer=tokenizer_partial, mask_token="<mask> ", collapse_mask_token=False)

    shap_values_list = []
    for idx, element in enumerate(prompt_list):
        split_element = element.split(" ")
        # only look if element is present
        if target_token in split_element:
            # get positions of elements
            positions = [index for index, value in enumerate(split_element) if value == target_token]

            # increment through each position is found in
            for pos in positions:
                # create partial function, need to add pos+1 as output is one token shuffled from input
                f_partial = partial(f, model=model, device=device, tokenizer=tokenizer, max_seq_length=max_seq_length, pad_token=pad_token, mask_token=mask_token, pos=pos + 1, encoder_only=encoder_only)
                
                # set max_evals to be same as permutations required for position to be masked and unmasked
                explainer = shap.PartitionExplainer(f_partial, masker, output_names=labels, max_evals= 2 * min(len(split_element), max_seq_length) + 1, seed=seed)

                # change indices to masks one at a time to ensure token doesn't imapct on itself
                split_element[pos] = "<mask>"
                new_element = " ".join(split_element)

                shap_values = explainer([new_element])

                # shap_values has three class objects:
                # .values: of shape (1, N_positions, N_token_ids)
                # .base_values: of shape (1, N_token_ids)
                # .data: list of all input data tokens
                # to get the shap value for a given position X on the token of interest Y,
                # need to get .base_values[Y] + .values[X, Y]

                #print(shap_values.values.shape)
                #print(shap_values.base_values.shape)

                #print(shap_values)

                # generate output array, concatenate base values to each row
                #output_array = (shap_values.values + shap_values.base_values).squeeze(0).T
                #print(output_array.shape)

                df = pd.DataFrame(shap_values.values.squeeze(0).T, index=labels, columns=split_element)
                df["base_value"] = shap_values.base_values.squeeze(0).T

                # Display the DataFrame
                df.to_csv(outpref + "_geneid_" + str(target_token) + "_fileidx_" + str(idx) + "_pos_" + str(pos) + ".csv", index=True)

def read_prompt_file(file_path):
    prompt_list = []
    with open(file_path, 'r') as file:
        for line in file:
            prompt_list.append(line.strip())
    return prompt_list

def query_model(rank, model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, encoder_only, target_token):
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

    calculate_SHAP(model, tokenizer, prompt_list, device, args.max_seq_length, encoder_only, target_token, args.outpref, args.seed)

        
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

    if DDP_active:
        #prompt_list = split_prompts(prompt_list, world_size)
        with Manager() as manager:
            mp.spawn(query_model,
                    args=(args.model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, args.encoder_only, args.target_token),
                    nprocs=world_size,
                    join=True)
    else:
        query_model(device, args.model_path, 1, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, args.encoder_only, args.target_token)
    

    if DDP_active:
        cleanup()

if __name__ == "__main__":
    main()