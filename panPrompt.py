import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import LEDConfig, LEDForConditionalGeneration, LEDTokenizer, logging
import numpy as np
import argparse
from tqdm import tqdm
import re

logging.set_verbosity_error()

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
def tokenize_prompt(prompt, max_seq_length, tokenizer):
    mask_token = tokenizer.mask_token_id

    encoded = tokenizer.encode(prompt)

    # merge consecutive masks into single mask token
    encoder_input = ' '.join([str(i) for i in encoded])
    #print('encoder_input pre merging')
    #print(encoder_input)
    pattern = f'({mask_token} )+'
    encoder_input = re.sub(pattern, str(mask_token) + ' ', encoder_input)
    pattern = f'( {mask_token})+'
    encoder_input = re.sub(pattern, ' ' + str(mask_token), encoder_input)

    encoder_input = [int(i) for i in encoder_input.split()]

    #print('encoder_input post merging')
    #print(encoder_input)

    encoder_input, attention_mask = pad_blocks(encoder_input, max_seq_length, pad_token=tokenizer.pad_token_id)

    #print('encoder_input post padding')
    #print(encoder_input)

    return encoder_input, attention_mask

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

def predict_next_tokens_BART(model, tokenizer, input_ids, attention_mask, device, batch_size, temperature):
    model.eval()

    output = ""
    num_batches = len(input_ids) // batch_size + (1 if len(input_ids) % batch_size != 0 else 0)

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, len(input_ids))

        # Stack input_ids and attention_mask for the current batch
        batch_input_ids = torch.cat(input_ids[start_index:end_index], dim=0).to(device)
        batch_attention_mask = torch.cat(attention_mask[start_index:end_index], dim=0).to(device)

        # Ensure all tokens attend globally just to the first token if first batch
        global_attention_mask = torch.zeros(batch_input_ids.shape, dtype=torch.long, device=batch_input_ids.device)
        if batch_index == 0:
            global_attention_mask[0, 0] = 1
        
        #print(batch_attention_mask)
        #print(batch_input_ids)

        # Generate summaries for the current batch
        summary_ids = model.generate(
            batch_input_ids,
            global_attention_mask=global_attention_mask,
            attention_mask=batch_attention_mask,
            # is max length here correct?
            max_length=batch_input_ids.shape[1],
            temperature=temperature,
            do_sample=True
        )

        # Decode the generated summaries
        for summary in summary_ids:
            decoded = tokenizer.decode(summary.tolist(), skip_special_tokens=True)
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
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for prediction.")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers.")
    parser.add_argument("--max_seq_length", type=int, default=16384, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Maximum batch size for simulation. Default = 16")
    parser.add_argument("--device", type=str, default='cpu', help="Device to run the model on (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--attention_window", type=int, default=512, help="Attention window size in the Longformer model (default: 512)")
    parser.add_argument("--prop_masked", type=float, default=0.3, help="Proportion of prompt to be masked. Default = 0.3")
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
            #print(tokens)
            input_ids = [torch.tensor([input]) for input in tokens]
            attention_mask = [torch.tensor([input], dtype=torch.long) for input in attention_mask]
            
            #print(prompt)
            for _ in range(num_seq):
                predicted_text = predict_next_tokens_BART(model, tokenizer, input_ids, attention_mask, device, args.batch_size, args.temperature)

                f.write(predicted_text + "\n")


if __name__ == "__main__":
    main()
