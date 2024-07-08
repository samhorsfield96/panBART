import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
import math
import torch.nn.functional as F
#from panGPT import SimpleTransformerModel, SimpleReformerModel
from transformers import LEDConfig, LEDForConditionalGeneration, LEDTokenizer, logging
import numpy as np
import argparse
from tqdm import tqdm

logging.set_verbosity_error()

# class PositionalEncoding(nn.Module):
#     """
#     Implements positional encoding as described in the Transformer paper.

#     Args:
#         d_model (int): The dimension of the embeddings (also called the model dimension).
#         dropout (float): Dropout rate.
#         max_len (int): Maximum length of the input sequences.

#     This module injects some information about the relative or absolute position of
#     the tokens in the sequence to make use of the order of the sequence.
#     """

#     def __init__(self, d_model, dropout=0.1, max_len=None):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         x = x + self.pe[: x.size(0), :]
#         return self.dropout(x)

# class SimpleTransformerModel(nn.Module):
#     """
#     A simple Transformer model for sequence generation.

#     Args:
#         vocab_size (int): Size of the vocabulary.
#         embed_dim (int): Dimension of the embedding layer.
#         num_heads (int): Number of attention heads in the transformer.
#         num_layers (int): Number of layers (stacks) in the transformer.
#         max_seq_length (int): Maximum length of the input sequences.
#         dropout_rate (float): Dropout rate in the transformer.

#     The model consists of an embedding layer, positional encoding, and a transformer encoder.
#     """

#     def __init__(
#         self,
#         vocab_size,
#         embed_dim,
#         num_heads,
#         num_layers,
#         max_seq_length,
#         dropout_rate=0.5,
#     ):
#         super(SimpleTransformerModel, self).__init__()
#         self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout_rate)
#         self.vocab_size = vocab_size
#         self.embed = nn.Embedding(vocab_size, embed_dim)
#         transformer_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=num_heads, dropout=dropout_rate
#         )
#         self.transformer = nn.TransformerEncoder(
#             transformer_layer, num_layers=num_layers
#         )
#         self.out = nn.Linear(embed_dim, vocab_size)

#     def forward(self, x):
#         x = self.embed(x)
#         x = self.transformer(x)
#         return self.out(x)

def pad_blocks(input_list, block_size, pad_token="<pad>"):
    # Initialize the result list
    result = []
    attention_mask = []

    # Iterate over the list in steps of block_size
    for i in range(0, len(input_list), block_size):
        # Get the current block
        block = input_list[i:i + block_size]
        attention_block = [1] * len(block)
        
        # Check if the block needs padding
        if len(block) < block_size:
            # Pad the block to the required block size
            attention_block += [0] * (block_size - len(block))
            block += [pad_token] * (block_size - len(block))            
        
        # Add the block to the result
        result.append(block)
        attention_mask.append(attention_block)
    
    return result, attention_mask

# returns list first entry is encoded, second is attention mask
def tokenize_prompt(prompt, max_length, tokenizer):

    encoded = tokenizer.encode(prompt)
    encoded, attention_mask = pad_blocks(encoded, max_length)

    return encoded, attention_mask

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

def load_model(model_path, embed_dim, num_heads, num_layers, max_seq_length, device, vocab_size, attention_window):
    # Infer the vocab size from the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    #vocab_size = checkpoint['model_state_dict']['embed.weight'].size(0)

    # if model_type == 'transformer':
    #     model = SimpleTransformerModel(vocab_size, embed_dim, num_heads, num_layers, max_seq_length)
    # elif model_type == 'reformer':
    #     model = SimpleReformerModel(vocab_size, embed_dim, reformer_depth, reformer_buckets, reformer_hashes)
    # elif model_type == "BARTlongformer":
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
        attention_window = attention_window
    )
    model = LEDForConditionalGeneration(BARTlongformer_config)
    #else:
        #raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model


def load_tokenizer(tokenizer_path):
    #return Tokenizer.from_file(tokenizer_path)
    return LEDTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)

def mask_integers(string, prop_masked):   
    # Identify the indices of the integers in the list
    integer_indices = np.array(string.split())
    
    # Determine how many integers to mask
    num_to_mask = int(len(integer_indices) * prop_masked)
    
    # Randomly select indices to mask
    if num_to_mask > 0:
        indices_to_mask = np.random.choice(range(len(integer_indices)), size=num_to_mask, replace=False)
    else:
        indices_to_mask = np.empty(shape=[0, 0])
    
    # Replace selected indices with "[MASK]"
    integer_indices[indices_to_mask] = "<mask>"
    
    # Reconstruct the string
    masked_string = ' '.join(integer_indices.tolist())
    
    return masked_string

# def predict_next_tokens(model, tokenizer, tokens, num_tokens, temperature=1.0):
#     model.eval()
#     for _ in range(num_tokens):
#         input_ids = torch.tensor([tokens])
#         with torch.no_grad():
#             outputs = model(input_ids)
#         scaled_logits = outputs[0, -1, :] / temperature
#         probabilities = F.softmax(scaled_logits, dim=-1)
#         next_token_id = torch.multinomial(probabilities, 1).item()
#         tokens.append(next_token_id)
#     return tokenizer.decode(tokens)

def predict_next_tokens_BART(model, tokenizer, input_ids, attention_mask, device, temperature=1.0):
    model.eval()

    output = ""

    for index, entry in enumerate(input_ids):
        entry = entry.to(device)
        mask = attention_mask[index].to(device)
    
        # ensure all tokens attend globally just to first token if first entry
        global_attention_mask = torch.zeros(entry.shape, dtype=torch.long, device=input_ids.device)
        if index == 0:
            global_attention_mask[:,[0]] = 1
        
        summary_ids = model.generate(entry, global_attention_mask=global_attention_mask, attention_mask=attention_mask, max_length=len(entry), temperature=temperature, do_sample=True)[0].tolist()
        decoded = tokenizer.decode(summary_ids, skip_special_tokens=True)
        output += decoded

    return output

def read_prompt_file(file_path):
    prompt_list = []
    with open(file_path, 'r') as file:
        for line in file:
            prompt_list.append(line.strip())
    return prompt_list

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Token prediction with a Transformer or Reformer model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--model_type", type=str, required=True, choices=['transformer', 'reformer', 'BARTlongformer'], help="Type of model (transformer or reformer).")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer file.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the text file containing the prompt.")
    parser.add_argument("--num_tokens", type=int, default=100, help="Number of tokens to predict. Default = 100")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for prediction.")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--device", type=str, default='cpu', help="Device to run the model on (e.g., 'cpu' or 'cuda').")
    #parser.add_argument("--max_len", type=int, default=5000, help="Maximum length for positional encoding.")
    parser.add_argument("--attention_window", type=int, default=512, help="Attention window size in the Longformer model (default: 512)")
    #parser.add_argument("--reformer_depth", type=int, default=6, help="Depth of the Reformer model.")
    #parser.add_argument("--reformer_buckets", type=int, default=32, help="Number of buckets in the Reformer model.")
    #parser.add_argument("--reformer_hashes", type=int, default=4, help="Number of hashes in the Reformer model.")
    parser.add_argument("--prop_masked", type=float, default=0, help="Proportion of prompt to be masked. Default = 0")
    parser.add_argument("--num_seq", type=int, default=1, help="Number of simulations per prompt sequence. Default = 1")
    parser.add_argument("--outfile", type=str, default="simulated_genomes.txt", help="Output file for simulated genomes. Default = 'simulated_genomes.txt'")
    args = parser.parse_args()

    args.max_seq_length = max(args.max_seq_length, args.attention_window)
    # Round down max_seq_length to the nearest multiple of attention_window
    args.max_seq_length = (args.max_seq_length // args.attention_window) * args.attention_window

    device = torch.device(args.device)

    prop_masked = args.prop_masked
    num_seq = args.num_seq

    tokenizer = load_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size

    model = load_model(args.model_path, args.embed_dim, args.num_heads, args.num_layers,
                        args.max_seq_length, device, vocab_size, args.attention_window)
    #if args.model_type == 'transformer':
    #    model.pos_encoding.pe = model.pos_encoding.pe[:args.max_len, :].to(device)  # Adjust the positional encoding based on max_len and device

    prompt_list = read_prompt_file(args.prompt_file)
    #if args.model_type == "BARTlongformer":
        
    with open(args.outfile, "w") as f:
        for prompt in tqdm(prompt_list, desc="Prompt number", total=len(prompt_list)):
            if prop_masked > 0:
                prompt = mask_integers(prompt, prop_masked)

            tokens, attention_mask = tokenize_prompt(prompt, args.max_seq_length, tokenizer)

            #tokens = tokenizer.encode(prompt)
            input_ids = [torch.tensor([input]) for input in tokens]
            attention_mask = [torch.tensor([input], dtype=torch.long) for input in attention_mask]
            
            #print(prompt)
            for _ in range(num_seq):
                #if args.model_type == "BARTlongformer":
                predicted_text = predict_next_tokens_BART(model, tokenizer, input_ids, attention_mask, device, args.temperature)
                # else:
                #     #prompt = original_prompt
                #     predicted_text = predict_next_tokens(model, tokenizer, tokens, args.num_tokens, args.temperature)
                #print("Prompt:", prompt)
                #print("Predicted text:", predicted_text)
                f.write(predicted_text + "\n")


if __name__ == "__main__":
    main()
