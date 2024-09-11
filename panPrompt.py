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
from panGPT import setup, cleanup, load_dataset
import random
from panGPT import mask_integers
from torch.utils.data import DataLoader, DistributedSampler

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
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for prediction.")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers.")
    parser.add_argument("--max_seq_length", type=int, default=16384, help="Maximum sequence length.")
    parser.add_argument("--model_dropout_rate", type=float, default=0.2, help="Dropout rate for the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Maximum batch size for simulation. Default = 16")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--attention_window", type=int, default=512, help="Attention window size in the Longformer model (default: 512)")
    parser.add_argument("--prop_masked", type=float, default=0.15, help="Proportion of prompt to be masked. Default = 0.15")
    parser.add_argument("--max_input_len", type=int, default=None, help="Maximum length of input sequence. No limit if not set.")
    parser.add_argument("--min_input_len", type=int, default=None, help="Minimum length of input sequence. No limit if not set.")
    parser.add_argument("--num_seq", type=int, default=1, help="Number of simulations per prompt sequence. Default = 1")
    parser.add_argument("--outfile", type=str, default="simulated_genomes.txt", help="Output file for simulated genomes. Default = 'simulated_genomes.txt'")
    parser.add_argument("--DDP", action="store_true", default=False, help="Multiple GPUs used via DDP during training.")

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

class GenomeDataset(torch.utils.data.Dataset):
    """
    Dataset class for genomic data.

    This class represents a dataset of genomic sequences for training, validation, or testing.

    Args:
    - texts (list): List of genomic sequences.
    - tokenizer (Tokenizer): Tokenizer for encoding genomic sequences.
    - max_length (int): Maximum length of the input sequence.

    Methods:
    - __len__(): Get the length of the dataset.
    - __getitem__(idx): Get an item from the dataset by index.
    """

    def __init__(self, texts, tokenizer, max_length, prop_masked):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.prop_masked = prop_masked
        self.mask_token = self.tokenizer.encode("<mask>").ids[0]
        self.pad_token = self.tokenizer.encode("<pad>").ids[0]
        self.bos_token = self.tokenizer.encode("<s>").ids[0]
        self.eos_token = self.tokenizer.encode("</s>").ids[0]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        input = self.tokenizer.encode(text).ids

        # generate decoder and labels input, wrapping decoder input to right
        labels = input[1:]

        # mask original text
        text = self.tokenizer.decode(labels, skip_special_tokens=False)
        text_masked = mask_integers(text, self.prop_masked)

        encoder_input = self.tokenizer.encode(text_masked).ids
        decoder_input = input[:-1]

        len_decoder = len(decoder_input)
        decoder_input = pad_input(decoder_input, self.max_length, self.pad_token)

        decoder_attention_mask = torch.ones(len(decoder_input), dtype=torch.long)
        decoder_attention_mask[len_decoder:] = 0

        labels = pad_input(labels, self.max_length, self.pad_token, labels=True)

        # merge consecutive masks into single mask token
        encoder_input = ' '.join([str(i) for i in encoder_input])
        #print('encoder_input pre merging')
        #print(encoder_input)
        pattern = f'({self.mask_token} )+'
        encoder_input = re.sub(pattern, str(self.mask_token) + ' ', encoder_input)
        pattern = f'( {self.mask_token})+'
        encoder_input = re.sub(pattern, ' ' + str(self.mask_token), encoder_input)
        #print('encoder_input post merging')
        #print(encoder_input)
        encoder_input = [int(i) for i in encoder_input.split()]

        len_masked = len(encoder_input)
        encoder_input = pad_input(encoder_input, self.max_length, self.pad_token)

        #print('encoder_input post padding')
        #print(encoder_input)

        # do not attend to mask tokens
        #print(int(self.mask_token))
        #mask_idx = np.flatnonzero(np.array(encoder_input) == int(self.mask_token))

        encoder_attention_mask = torch.ones(len(encoder_input), dtype=torch.long)
        encoder_attention_mask[len_masked:] = 0
        #encoder_attention_mask[mask_idx] = 0

        # attend to all contig breaks
        global_attention_mask = torch.zeros(len(encoder_input), dtype=torch.long)
        break_idx = np.flatnonzero(np.array(encoder_input) == int(self.tokenizer.encode("_").ids[0]))
        global_attention_mask[break_idx] = 1

        # print("labels")
        # print(len(labels))
        # print(labels)
        # print("decoder_input")
        # print(len(decoder_input))
        # print(decoder_input)
        # print("decoder_attention_mask")
        # print(len(decoder_attention_mask.tolist()))
        # print(decoder_attention_mask.tolist())
        # print("encoder_input")
        # print(len(encoder_input))
        # print(encoder_input)
        # print("encoder_attention_mask")
        # print(len(encoder_attention_mask.tolist()))
        # print(encoder_attention_mask.tolist())
        # print("global_attention_mask")
        # print(len(global_attention_mask.tolist()))
        # print(global_attention_mask.tolist())

        return torch.tensor(decoder_input, dtype=torch.long), torch.tensor(encoder_input, dtype=torch.long), torch.tensor(labels, dtype=torch.long), decoder_attention_mask, encoder_attention_mask, global_attention_mask

def print_banner():
    banner = '''
    **************************************************
    *                                                *
    *        Transformer Model Token Prediction      *
    *        panPrompt v0.01                         *
    *        author: James McInerney                 *
    *                                                *
    **************************************************
    '''
    print(banner)

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
    return model

def predict_next_tokens_BART(model, tokenizer, prompt, loader, device, batch_size, temperature, prop_masked, num_seq, max_seq_length, DDP_active):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    vocab_size = tokenizer.get_vocab_size()
    pad_token = tokenizer.encode("<pad>").ids[0]

    generate = False

    total_val_loss = 0
    total_accuracy = 0
    preds_all = []
    labels_all = []
    predictions = []
    with torch.no_grad():
        for decoder_input, encoder_input, labels, decoder_attention_mask, encoder_attention_mask, global_attention_mask in loader:  # Correctly unpack the tuples returned by the DataLoader
  
            total_len = encoder_input.size(1)
            current_sequence = []
            for i in range(0, total_len, max_seq_length):
                print("max_seq_length exceeded: {}".format(total_len > max_seq_length))

                batch_decoder_input, batch_encoder_input, batch_decoder_attention_mask, batch_encoder_attention_mask, batch_global_attention_mask = decoder_input[:, i:i + max_seq_length].to(device), encoder_input[:, i:i + max_seq_length].to(device), decoder_attention_mask[:, i:i + max_seq_length].to(device), encoder_attention_mask[:, i:i + max_seq_length].to(device), global_attention_mask[:, i:i + max_seq_length].to(device) # Move data to the appropriate device

                if generate == True:
                    if DDP_active:
                        summary_ids = model.module.generate(
                            batch_encoder_input,
                            global_attention_mask=batch_global_attention_mask,
                            attention_mask=batch_encoder_attention_mask,
                            # # is max length here correct?
                            max_length=batch_encoder_input.shape[1],
                            temperature=1.0,
                            #num_beams=4,             # Beam size (the number of beams to explore)
                            #early_stopping=True,     # Stop when the first <eos> token is generated
                            num_return_sequences=num_seq,   # Number of sequences to return
                            #top_p=0.9,
                            #repetition_penalty=1.0,
                            do_sample=True,
                            decoder_start_token_id=batch_decoder_input[:, 0]
                        )
                    else:
                        summary_ids = model.generate(
                            batch_encoder_input,
                            global_attention_mask=batch_global_attention_mask,
                            attention_mask=batch_encoder_attention_mask,
                            # # is max length here correct?
                            max_length=batch_encoder_input.shape[1],
                            temperature=1.0,
                            #num_beams=4,             # Beam size (the number of beams to explore)
                            #early_stopping=True,     # Stop when the first <eos> token is generated
                            num_return_sequences=num_seq,   # Number of sequences to return
                            #top_p=0.9,
                            #repetition_penalty=1.0,
                            do_sample=True,
                            decoder_start_token_id=batch_decoder_input[:, 0]
                        )

                    batch_labels = labels[:, i:i + max_seq_length].to(device)

                    # Decode the generated summaries
                    for index, preds in enumerate(summary_ids):

                        # ignore padding positions
                        mask = batch_labels != -100
                        preds = preds[mask[0]]
                        batch_labels = batch_labels[mask]

                        # calculate accuracy
                        correct = (preds[1:] == batch_labels[:-1]).sum().item()
                        accuracy = correct / batch_labels[:-1].numel()
                        total_accuracy += accuracy * batch_labels[:-1].size(0)  # Accumulate the accuracy

                        print("accuracy:")
                        print(accuracy)
                        print("preds:")
                        print(preds[1:].tolist())
                        print("labels:")
                        print(batch_labels[:-1].tolist())
                        print("matches:")
                        print((preds[1:] == batch_labels[:-1]).type(torch.uint8).tolist())

                        current_sequence.append(preds)
                        
                        fail

                    # decoded = tokenizer.decode(summary.tolist()[0: len(encoded_blocks[index])], skip_special_tokens=True)
                    # #print("decoded:")
                    # #print(decoded)
                    # output_seqs[index] += decoded

                else:
                    outputs = model(input_ids=batch_encoder_input, attention_mask=batch_encoder_attention_mask, decoder_input_ids=batch_decoder_input, decoder_attention_mask=batch_decoder_attention_mask, global_attention_mask=batch_global_attention_mask).logits  # Generate predictions
                    #outputs = model(input_ids=encoder_input, attention_mask=encoder_attention_mask, decoder_input_ids=decoder_input, decoder_attention_mask=decoder_attention_mask).logits

                    # Free GPU memory
                    del encoder_input
                    del encoder_attention_mask
                    del decoder_input
                    del decoder_attention_mask
                    del global_attention_mask

                    #torch.cuda.empty_cache()

                    batch_labels = labels[:, i:i + max_seq_length].to(device)
                    
                    loss = criterion(outputs.view(-1, vocab_size), batch_labels.view(-1))
                    total_val_loss += loss.item() * batch_labels.size(0)  # Accumulate the loss

                    preds = outputs.argmax(dim=-1)  # Get predicted classes

                    # ignore padding positions
                    mask = batch_labels != -100
                    preds = preds[mask]
                    batch_labels = batch_labels[mask]

                    # calculate accuracy
                    correct = (preds == batch_labels).sum().item()
                    accuracy = correct / batch_labels.numel()
                    total_accuracy += accuracy * batch_labels.size(0)  # Accumulate the accuracy

                    print("accuracy:")
                    print(accuracy)
                    print("loss:")
                    print(loss.item())
                    # print("preds:")
                    # print(preds.tolist())
                    # print("labels:")
                    # print(batch_labels.tolist())
                    print("matches:")
                    print((preds == batch_labels).type(torch.uint8).tolist())

                    print(preds)
                    current_sequence.append(preds)
            
                # concatenate predictions
                current_sequence = [tensor.unsqueeze(0) if tensor.ndim == 1 else tensor for tensor in current_sequence]
                sequence = torch.cat(current_sequence, dim=1)
                predictions.append(sequence)
            print(predictions)

    return predictions

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

def query_model(rank, model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, return_list):
    if DDP_active:
        setup(rank, world_size)
        #prompt_list = prompt_list[rank]
        sampler = DistributedSampler(prompt_list, num_replicas=world_size, rank=rank, shuffle=True)
        num_workers = 0
        pin_memory = False
        shuffle = False
    else:
        sampler = None
        pin_memory = True
        shuffle = True
        num_workers=1
    
    dataset = GenomeDataset(prompt_list, tokenizer, args.max_seq_length, args.prop_masked)
    dataset.attention_window = args.attention_window
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
    dataset_size = len(loader.dataset)
    
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
    #for prompt in tqdm(prompt_list, desc="Prompt number", total=len(prompt_list), disable=not master_process and DDP_active):

    predicted_text = predict_next_tokens_BART(model, tokenizer, prompt_list[0], loader, device, args.batch_size, args.temperature, args.prop_masked, args.num_seq, args.max_seq_length, DDP_active)
    #print(predicted_text)
    return_list.append(predicted_text)

        
def main():
    print_banner()
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

    return_list = []
    if DDP_active:
        prompt_list = split_prompts(prompt_list, world_size)
        with Manager() as manager:
            mp_list = manager.list()
            mp.spawn(query_model,
                    args=(args.model_path, world_size, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, mp_list),
                    nprocs=world_size,
                    join=True)
            return_list = list(mp_list)
    else:
        query_model(device, args.model_path, 1, args, BARTlongformer_config, tokenizer, prompt_list, DDP_active, return_list)
    
    with open(args.outfile, "w") as f:
        for entry in return_list:
            f.write(entry + "\n")

    if DDP_active:
        cleanup()

if __name__ == "__main__":
    main()
