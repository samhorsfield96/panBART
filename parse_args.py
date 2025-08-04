import argparse

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import LEDConfig, LEDForConditionalGeneration, LEDTokenizer, logging
import numpy as np
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
from panGPT import GenomeDataset
from collections import defaultdict
from torch.nn.functional import cosine_similarity
import pandas as pd
import sys
import re

# Command line argument parsing
def parse_args():
    """
    Parse command-line arguments.

    This function parses the command-line arguments provided by the user and returns
    a Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a transformer model on (pan)'omic data.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to the input file")
    parser.add_argument("--train_file", type=str, default=None, help="Path to the training input file")
    parser.add_argument("--val_file", type=str, default=None, help="Path to the validation input file")
    parser.add_argument("--test_file", type=str, default=None, help="Path to the test input file")
    parser.add_argument("--attention_window", type=int, default=512, help="Attention window size in the Longformer model (default: 512)")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--max_seq_length", type=int, default=16384, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training and validation. This is per GPU.")
    parser.add_argument("--model_dropout_rate", type=float, default=0.2, help="Dropout rate for the model")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5, help="Factor by which the learning rate will be reduced by the learning rate scheduler")
    parser.add_argument("--lr_patience", type=int, default=10, help="Patience for learning rate reduction")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=0.01, help="Minimum delta for early stopping")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max_vocab_size", type=int, default=None, help="Maximum vocabulary size. Tokens beyond this size will be mapped to <UNK>. If not set, will infer maximum, although may lead to OOM issue.")
    parser.add_argument("--model_save_path", type=str, default="./model_checkpoint.pth", help="Path to save the model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="./pangenome_gpt_tokenizer", help="Path for saving and loading the tokenizer.")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of the dataset to include in the training set")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of the dataset to include in the validation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pe_dropout_rate", type=float, default=0.1, help="Dropout rate for positional encoding")
    parser.add_argument("--max_input_len", type=int, default=None, help="Maximum length of input sequence. No limit if not set.")
    parser.add_argument("--min_input_len", type=int, default=None, help="Minimum length of input sequence. No limit if not set.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save TensorBoard logs")
    parser.add_argument("--device", default=None, help="GPU device number if available. If not specified, will use all available Default = None")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of threads for data loading. Default = 1")
    parser.add_argument("--prop_masked", type=float, default=0.3, help="Average proportion of inputs to be masked. Default = 0.3")
    parser.add_argument("--restart", default=False, action="store_true", help="Restart model if checkpoint file present.")
    parser.add_argument("--reuse_tokenizer", default=False, action="store_true", help="Reuse existing tokenizer if present.")
    parser.add_argument("--gradient_checkpointing", default=False, action="store_true", help="Use gradient checkpointing during training. Improves memory efficiency at cost to runtime.")
    parser.add_argument("--encoder_only", default=False, action="store_true", help="Train using encoder input only.")
    parser.add_argument("--save_test_data", default=False, action="store_true", help="Print the genomes used for testing as held-out sequences.")
    parser.add_argument("--global_contig_breaks", default=False, action="store_true", help="Attend globally to contig breaks. Default is local only.")
    parser.add_argument("--port", default="12356", type=str, help="GPU port for DDP. Default=12356")
    
    args = parser.parse_args()

    # Ensure max_seq_length is greater than or equal to attention_window
    args.max_seq_length = max(args.max_seq_length, args.attention_window)
    # Round down max_seq_length to the nearest multiple of attention_window
    args.max_seq_length = (args.max_seq_length // args.attention_window) * args.attention_window


    return args