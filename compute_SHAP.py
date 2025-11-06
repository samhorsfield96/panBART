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
from panGPT import setup, cleanup, GenomeDataset
import random
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
import shap
import scipy as sp
import pandas as pd

logging.set_verbosity_error()

def read_prompt_file(file_path):

    # ensure genomes are placed in the same order and are present
    genome_id_list = []
    prompt_list = []
    with open(file_path, 'r') as file:
        for line in file:
            split_line = line.strip().split("\t")
            genome_id_list.append(split_line[0])
            prompt_list.append(split_line[1])
    
    return prompt_list, genome_id_list

def parse_args():
    """
    Parse command-line arguments.

    This function parses the command-line arguments provided by the user and returns
    a Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Token prediction with a Transformer or Reformer model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer file.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the text file containing the prompt, must contain genome ids and genomes in tab separated format.")
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
    parser.add_argument("--randomise", action="store_true", default=False, help="Randomise input tokens.")
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


# this defines an explicit python function that takes a list of strings and outputs scores for each class
def f(x, model, device, tokenizer, max_seq_length, pos, args, encoder_only=False):
    outputs = []
    model.eval()

    #print(f"x: {x}")
    dataset = GenomeDataset(x, tokenizer, args.max_seq_length, 0, args.global_contig_breaks, False)
    dataset.attention_window = args.attention_window
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=False, sampler=None)
    with torch.no_grad():
        for _x in loader:
            #print(f"_x: {_x}")
            decoder_input, encoder_input, labels, decoder_attention_mask, encoder_attention_mask, global_attention_mask = _x
            #print(f"encoder_input: {encoder_input}")
            #print(pos)
            if encoder_only:
                batch_encoder_input, batch_encoder_attention_mask, batch_global_attention_mask = encoder_input[:, 0:max_seq_length].to(device), encoder_attention_mask[:, 0:max_seq_length].to(device), global_attention_mask[:, 0:max_seq_length].to(device)
                output = model(input_ids=batch_encoder_input, attention_mask=batch_encoder_attention_mask, global_attention_mask=batch_global_attention_mask, output_hidden_states=True)  # Generate predictions
                encoder_hidden_states = output.encoder_last_hidden_state
                encoder_logits = model.module.lm_head(encoder_hidden_states) + model.module.final_logits_bias
                encoder_logits = encoder_logits.detach().cpu().numpy()
                outputs.append(encoder_logits[0][pos])
            else:
                batch_decoder_input, batch_encoder_input, batch_decoder_attention_mask, batch_encoder_attention_mask, batch_global_attention_mask = decoder_input[:, 0:max_seq_length].to(device), encoder_input[:, 0:max_seq_length].to(device), decoder_attention_mask[:, 0:max_seq_length].to(device), encoder_attention_mask[:, 0:max_seq_length].to(device), global_attention_mask[:, 0:max_seq_length].to(device)
                output = model(input_ids=batch_encoder_input, attention_mask=batch_encoder_attention_mask, decoder_input_ids=batch_decoder_input, decoder_attention_mask=batch_decoder_attention_mask, global_attention_mask=batch_global_attention_mask).logits.detach().cpu().numpy()
            
                #print(output)
                #decoder shifted one to right, so adjust accordingly
                outputs.append(output[0][pos - 1])
            
    torch.cuda.empty_cache()

    # save all scores in same output
    #outputs = output[0]
    outputs = np.array(outputs)
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores)
    #print(val)
    return val

#may need to change this to use the correct tokenizer
class CustomTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, s):
        #print(f"self.tokenizer.encode(s).ids: {self.tokenizer.encode(s).ids}")
        return {"input_ids": self.tokenizer.encode(s).ids}

    def decode(self, a):
        #print(f"self.tokenizer.decode(a, skip_special_tokens=False): {self.tokenizer.decode(a, skip_special_tokens=False)}")
        return self.tokenizer.decode(a, skip_special_tokens=False)

def calculate_SHAP(model, tokenizer, prompt_list, device, max_seq_length, encoder_only, target_token, outpref, seed, args, genome_labels=None, both_strands=False):
    # follow this example https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/sentiment_analysis/Using%20custom%20functions%20and%20tokenizers.html

    # Get the vocabulary dictionary: {token: token_id}
    vocab_dict = tokenizer.get_vocab()

    # Sort tokens by token ID to get them in the order they appear in the logits
    sorted_vocab = sorted(vocab_dict.items(), key=lambda item: item[1])

    # Extract the tokens in order of their IDs
    labels = [token for token, _ in sorted_vocab]
    #print(labels)

    # create partial tokenizer
    custom_tokenizer = CustomTokenizer(tokenizer)
    # issue might be how masker is working, if it truncates sequences rather than just masking positions.
    masker = shap.maskers.Text(tokenizer=custom_tokenizer, mask_token="<mask>", collapse_mask_token=False)

    shap_values_list = []
    for idx, element in enumerate(prompt_list):
        #print(f"element: {element}")
        split_element = element.split(" ")
        #print(f"split_element: {split_element}")
        # only look if element is present

        # look for both strands of gene
        if both_strands:
            target_token_tup = (target_token, str(int(target_token) * -1))
        else:
            target_token_tup = (target_token)

        for token in target_token_tup:
            if token in split_element:
                # get positions of elements
                positions = [index for index, value in enumerate(split_element) if value == token]

                # increment through each position is found in
                for pos in positions:
                    # need to add pos+1 as output is one token shuffled from input
                    # if encoder_only == False:
                    #     target_pos = pos + 1
                    # else:
                    #     target_pos = pos
                    target_pos = pos
                    
                    # create partial function
                    f_partial = partial(f, model=model, device=device, tokenizer=tokenizer, max_seq_length=max_seq_length, pos=target_pos, args=args, encoder_only=encoder_only)
                    
                    # set max_evals to be same as permutations required for position to be masked and unmasked
                    explainer = shap.PartitionExplainer(f_partial, masker, output_names=labels, max_evals= 2 * min(len(split_element), max_seq_length) + 1, seed=seed)

                    # change indices to masks one at a time to ensure token doesn't impact on itself
                    new_element = split_element
                    new_element[pos] = "<mask>"
                    new_element = " ".join(new_element)

                    #print(f"new_element: {new_element}")

                    # dataset = GenomeDataset([new_element], tokenizer, args.max_seq_length, 0, args.global_contig_breaks, False)
                    # dataset.attention_window = args.attention_window
                    # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=False, sampler=None)

                    # for tensor_set in loader:
                    #     shap_values = explainer([tensor_set])

                    shap_values = explainer([new_element])

                    # shap_values has three class objects:
                    # .values: of shape (1, N_positions, N_token_ids)
                    # .base_values: of shape (1, N_token_ids)
                    # .data: list of all input data tokens
                    # to get the shap value for a given position X on the token of interest Y,
                    # need to get .base_values[Y] + .values[X, Y]

                    #print(f"shap_values.values.shape: {shap_values.values.shape}")
                    #print(shap_values.base_values.shape)

                    #print(f"shap_values: {shap_values}")
                    #print(f"shap_values.data[0].shape: {shap_values.data[0].shape}")
                    #print(f"shap_values.data: {shap_values.data[0].tolist()}")
                    
                    #print(f"labels: {labels}")
                    #print(f"split_element: {split_element}")

                    # generate output array, concatenate base values to each row
                    #output_array = (shap_values.values + shap_values.base_values).squeeze(0).T
                    #print(output_array.shape)

                    df = pd.DataFrame(shap_values.values.squeeze(0).T, index=labels, columns=split_element)
                    df["base_value"] = shap_values.base_values.squeeze(0).T

                    # Display the DataFrame
                    if genome_labels != None:
                        df.to_csv(outpref + "_" + str(genome_labels[idx]) + "_geneid_" + str(token) + "_pos_" + str(pos) + ".csv", index=True)
                    else:
                        df.to_csv(outpref  + "_fileidx_" + str(idx) + "_geneid_" + str(token) + "_pos_" + str(pos) + ".csv", index=True)

def query_model(rank, model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, genome_labels, DDP_active, encoder_only, target_token):
    if DDP_active:
        setup(rank, world_size, args.port)
        
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

    calculate_SHAP(model, tokenizer, prompt_list, device, args.max_seq_length, encoder_only, target_token, args.outpref, args.seed, args, genome_labels, True)

        
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

    prompt_list, genome_labels = read_prompt_file(args.prompt_file)

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
        prompt_list_tup = [(genome, genome_id) for genome, genome_id in zip(prompt_list, genome_labels) if len(genome.split()) <= args.max_input_len]
        prompt_list = [entry[0] for entry in prompt_list_tup]
        genome_labels = [entry[1] for entry in prompt_list_tup]

    if args.min_input_len != None:
        # len_list = [len(genome.split()) for genome in prompt_list]
        # print(len_list)
        prompt_list_tup = [(genome, genome_id) for genome, genome_id in zip(prompt_list, genome_labels) if len(genome.split()) >= args.max_input_len]
        prompt_list = [entry[0] for entry in prompt_list_tup]
        genome_labels = [entry[1] for entry in prompt_list_tup]

    if DDP_active:
        #prompt_list = split_prompts(prompt_list, world_size)
        with Manager() as manager:
            mp.spawn(query_model,
                    args=(args.model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, genome_labels, DDP_active, args.encoder_only, args.target_token),
                    nprocs=world_size,
                    join=True)
    else:
        query_model(device, args.model_path, 1, args, BARTlongformer_config, tokenizer, prompt_list, genome_labels, DDP_active, args.encoder_only, args.target_token)
    

    if DDP_active:
        cleanup()

if __name__ == "__main__":
    main()