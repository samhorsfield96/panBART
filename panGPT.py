import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, ByteLevelBPETokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import random
import numpy as np
from tqdm import tqdm
from transformers import LEDConfig, LEDForConditionalGeneration, LEDTokenizer
from pathlib import Path
import random
import re

# Global variables
PROGRAM_NAME = "panGPT"
VERSION = "0.10a"
AUTHOR = "James McInerney"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("training.log"), 
        logging.StreamHandler()
    ],
)

def print_banner():
    """
    Print the program banner with information about the program name, version, and author.

    This function prints the banner with the program name, version, and author information,
    along with additional disclaimer and license details.
    """
    border_symbol = "="
    padding_outer = 5
    padding_inner = 3
    full_program_name = f"{PROGRAM_NAME} v{VERSION}"
    line_length = len(full_program_name) + 2 * padding_inner
    border_line = border_symbol * (line_length + 2 * padding_outer)

    print(border_line)
    print(f"{border_symbol * padding_outer}{' ' * padding_inner}{full_program_name}{' ' * padding_inner}{border_symbol * padding_outer}")
    print(border_line)
    print(f"Developed by: {AUTHOR}")
    print("Website: http://mcinerneylab.com/")
    print("DISCLAIMER: This code is provided 'AS IS' without any warranties of any kind, including,")
    print("but not limited to, its fitness for a particular purpose. The author disclaims all ")
    print("liability for any damages, direct, indirect, tangential, incidental or consequential, ")
    print("resulting from the use of this code.")
    print("Licensed under the GNU General Public License v3.0")
    print("Full license at: https://www.gnu.org/licenses/gpl-3.0.en.html")
    print(border_line)

print_banner()

def mask_integers(string, prop_masked):   
   
    # Randomly select indices to mask
    if prop_masked > 0:
        # Identify the indices of the integers in the list
        integer_indices = np.array(string.split())
        
        # Determine how many integers to mask
        num_to_mask = int(len(integer_indices) * prop_masked)

        # sample number sites from poisson
        num_to_mask = np.random.poisson(num_to_mask)
        indices_to_mask = np.random.choice(range(len(integer_indices)), size=num_to_mask, replace=False)

        # Replace selected indices with "[MASK]"
        integer_indices[indices_to_mask] = "<mask>"

        # Reconstruct the string
        masked_string = ' '.join(integer_indices.tolist())

        return masked_string
    else:
        return string    

def pad_input(input, max_length, tokenizer, labels=False):

    len_masked = len(input)
    if labels == False:
        input.extend([tokenizer.pad_token_id] * (max_length - len_masked))
    else:
        input.extend([-100] * (max_length - len_masked))

    return input

# Command line argument parsing
def parse_args():
    """
    Parse command-line arguments.

    This function parses the command-line arguments provided by the user and returns
    a Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a transformer model on (pan)'omic data.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    #parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "longformer", "BARTlongformer"], help="Type of model to use: 'transformer' or 'longformer'")
    parser.add_argument("--attention_window", type=int, default=512, help="Attention window size in the Longformer model (default: 512)")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--max_seq_length", type=int, default=16384, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training and validation")
    parser.add_argument("--model_dropout_rate", type=float, default=0.2, help="Dropout rate for the model")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5, help="Factor by which the learning rate will be reduced by the learning rate scheduler")
    parser.add_argument("--lr_patience", type=int, default=10, help="Patience for learning rate reduction")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=0.01, help="Minimum delta for early stopping")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max_vocab_size", type=int, default=70000, help="Maximum vocabulary size. Tokens beyond this size will be mapped to <UNK>.")
    parser.add_argument("--model_save_path", type=str, default="./model_checkpoint.pth", help="Path to save the model checkpoint")
    parser.add_argument("--tokenizer_dir", type=str, default="./pangenome_gpt_tokenizer", help="Dirname for saving and loading the tokenizer")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of the dataset to include in the training set")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of the dataset to include in the validation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pe_dropout_rate", type=float, default=0.1, help="Dropout rate for positional encoding")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save TensorBoard logs")
    parser.add_argument("--device", type=int, default=0, help="GPU device number if available. Default = 0")
    parser.add_argument("--prop_masked", type=float, default=0.3, help="Average proportion of inputs to be masked. Default = 0.3")
    parser.add_argument("--restart", default=False, action="store_true", help="Restart model if checkpoint file present.")
    
    args = parser.parse_args()

    # Ensure max_seq_length is greater than or equal to attention_window
    args.max_seq_length = max(args.max_seq_length, args.attention_window)
    # Round down max_seq_length to the nearest multiple of attention_window
    args.max_seq_length = (args.max_seq_length // args.attention_window) * args.attention_window


    return args

args = parse_args()
params = vars(args)  # Convert the parsed arguments to a dictionary

input_file = args.input_file
#model_type = args.model_type
attention_window=args.attention_window
embed_dim = args.embed_dim
num_heads = args.num_heads
num_layers = args.num_layers
max_seq_length = args.max_seq_length
batch_size = args.batch_size
model_dropout_rate = args.model_dropout_rate
learning_rate = args.learning_rate
lr_scheduler_factor = args.lr_scheduler_factor
weight_decay = args.weight_decay
lr_patience = args.lr_patience
early_stop_patience =args.early_stop_patience
min_delta = args.min_delta
epochs = args.epochs
max_vocab_size = args.max_vocab_size
model_save_path = args.model_save_path
tokenizer_dir = args.tokenizer_dir
train_size = args.train_size
val_size = args.val_size
seed = args.seed
#pe_max_len = args.pe_max_len
pe_dropout_rate = args.pe_dropout_rate
log_dir = args.log_dir
prop_masked = args.prop_masked
restart = args.restart

# Check if max_seq_length is a multiple of attention_window when using Longformer
#if (model_type == "longformer" or model_type == "BARTlongformer") and max_seq_length % attention_window != 0:
if max_seq_length % attention_window != 0:
    logging.info(f"Error: When using the LED model, the maximum sequence length (max_seq_length) must be a multiple of the attention window size (attention_window).")
    logging.info(f"Current values: max_seq_length = {max_seq_length}, attention_window = {attention_window}")
    logging.info("Please adjust these values and try again.")
    exit(1)


# Check whether certain files exist
if not os.path.isfile(input_file):
    print(f"Error: The specified input file '{input_file}' does not exist.")
    exit(1)
if model_save_path and not os.path.isdir(os.path.dirname(model_save_path)):
    print(f"Error: The directory for model save path '{model_save_path}' does not exist.")
    exit(1)

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
    - seed (int): The random seed value to set.

    This function sets the random seed for the random number generators in PyTorch,
    NumPy, and Python's built-in random module to ensure reproducibility of results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

def print_parameters_table(params: dict) -> None:
    """
    Prints a table of parameters and their settings.

    Args:
    - params (dict): Dictionary of parameters and their values.
    """
    try:
        max_param_length = max(len(param) for param in params)
        header = "-" * (max_param_length + 20)
        logging.info("\nParameters:")
        logging.info(header)
        logging.info("{:<{width}}: {}".format("Parameter", "Setting", width=max_param_length))
        logging.info(header)

        for param, value in params.items():
            if param == "attention_window":
                continue  # Skip this parameter unless 'longformer' model is selected
            logging.info("{:<{width}}: {}".format(param, value, width=max_param_length))

        logging.info(header)

    except Exception as e:
        logging.error(f"Failed to log parameters: {e}")

print_parameters_table(params)

def load_dataset(input_file):
    """
    Load the dataset from the input file.

    Args:
    - input_file (str): Path to the input file containing the dataset.

    Returns:
    - list: List of strings, each representing a genome sequence.

    This function reads the contents of the input file, which contains genome sequences,
    and returns a list of genome sequences.
    """
    try:
        with open(input_file, "r") as file:
            genomes = [genome.strip() for genome in file.readlines()]
        return genomes
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit(1)

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Save the model checkpoint to a file.

    Args:
    - model: The PyTorch model to save.
    - optimizer: The optimizer state associated with the model.
    - epoch (int): The current epoch number.
    - loss: The loss value at the current epoch.
    - save_path (str): Path to save the model checkpoint file.

    This function saves the model checkpoint, including the model state dictionary,
    optimizer state dictionary, current epoch number, and loss value, to the specified file.
    """

    try:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, save_path)
    except IOError as e:
        print(f"Failed to save checkpoint to '{save_path}': {e}")

def load_checkpoint(model, optimizer, checkpoint_path, restart):
    """
    Load a model checkpoint from a file.

    Args:
    - model: The PyTorch model to load the checkpoint into.
    - optimizer: The optimizer associated with the model.
    - checkpoint_path (str): Path to the model checkpoint file.

    Returns:
    - tuple: A tuple containing the start epoch number and a boolean indicating
             whether the checkpoint was successfully loaded.

    This function loads a model checkpoint from the specified file into the given model
    and optimizer. It returns the start epoch number and a boolean indicating whether
    the checkpoint was successfully loaded.
    """

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting from scratch.")
        return 0, False
    try:
        if restart:
            print("Restarting training, overwriting existing checkpoint.")
            return 0, False
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")
        return start_epoch, True
    except Exception as e:
        if "size mismatch" in str(e):
            error_msg = "Error: Checkpoint and current model do not match in size."
        else:
            error_msg = f"Error loading checkpoint from '{checkpoint_path}': {str(e)}"
        logging.error(error_msg)
        return 0, False

def train_model(train_loader, model, optimizer, criterion, device):
    """
    Train the transformer model on the training dataset.

    Args:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - model (nn.Module): Transformer model to train.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - criterion: Loss criterion for computing the loss.
    - device (torch.device): Device to perform computations on (CPU or GPU).

    Returns:
    - float: Average training loss.

    This function trains the transformer model on the training dataset for one epoch,
    computes the average training loss, and returns it.
    """

    model.train()  # Set the model to training mode
    total_train_loss = 0
    for i, (decoder_input, encoder_input, labels, decoder_attention_mask, encoder_attention_mask, beginning) in enumerate(train_loader):  # Added enumeration for clarity
        decoder_input, encoder_input, decoder_attention_mask, encoder_attention_mask = decoder_input.to(device), encoder_input.to(device), decoder_attention_mask.to(device), encoder_attention_mask.to(device)  # Move data to the appropriate device
        
        beginning = torch.nonzero(beginning)

        # set global attention for all tokens so that all positions attend to all others.
        global_attention_mask = torch.zeros(decoder_input.shape, dtype=torch.long, device=decoder_input.device)
        global_attention_mask[beginning,[0]] = 1
        
        optimizer.zero_grad()  # Clear gradients before calculating them
        outputs = model(input_ids=encoder_input, attention_mask=encoder_attention_mask, decoder_input_ids=decoder_input, decoder_attention_mask=decoder_attention_mask, global_attention_mask=global_attention_mask).logits  # Generate predictions
        
        # Free GPU memory
        del encoder_input
        del encoder_attention_mask
        del decoder_input
        del decoder_attention_mask
        del global_attention_mask

        labels = labels.to(device)
        
        loss = criterion(outputs.view(-1, model.config.vocab_size), labels.view(-1))
        loss.backward()  # Compute gradient of the loss w.r.t. network parameters
        optimizer.step()  # Update parameters based on gradient

        total_train_loss += loss.item() * labels.size(0)  # Accumulate the loss
        # Update the progress bar
        train_loader.set_description(f"Epoch {epoch} - Training")
        train_loader.update(1)

    avg_train_loss = total_train_loss / train_dataset_size
    return avg_train_loss

def calculate_metrics(preds, labels):
    """
    Calculate evaluation metrics for model predictions.

    Args:
    - preds (Tensor): Predicted labels.
    - labels (Tensor): True labels.

    Returns:
    - tuple: Tuple containing evaluation metrics (accuracy, precision, recall, F1 score, Cohen's kappa).

    This function calculates various evaluation metrics (accuracy, precision, recall, F1 score, Cohen's kappa)
    based on the predicted labels and true labels, and returns them as a tuple.
    """

    preds = preds.view(-1)
    labels = labels.view(-1)
    if torch.unique(preds).size(0) == 1:
        warnings.warn("All predicted labels are the same. The model might not be learning properly.")
    accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average="weighted", zero_division=0)
    recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average="weighted", zero_division=0)
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="weighted", zero_division=0)
    kappa = cohen_kappa_score(labels.cpu().numpy(), preds.cpu().numpy())
    return accuracy, precision, recall, f1, kappa

def validate_model(val_loader, model, criterion, device, epoch=None):
    """
    Validate the transformer model on the validation dataset.

    Args:
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - model (nn.Module): Transformer model to validate.
    - criterion: Loss criterion for computing the loss.
    - device (torch.device): Device to perform computations on (CPU or GPU).

    Returns:
    - tuple: Tuple containing validation metrics (average validation loss, accuracy,
             precision, recall, F1 score, Cohen's kappa).

    This function validates the transformer model on the validation dataset,
    computes various validation metrics (average validation loss, accuracy, precision,
    recall, F1 score, Cohen's kappa), and returns them as a tuple.
    """

    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    total_accuracy = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for decoder_input, encoder_input, labels, decoder_attention_mask, encoder_attention_mask, beginning in val_loader:  # Correctly unpack the tuples returned by the DataLoader
            decoder_input, encoder_input, decoder_attention_mask, encoder_attention_mask = decoder_input.to(device), encoder_input.to(device), decoder_attention_mask.to(device), encoder_attention_mask.to(device)  # Move data to the appropriate device

            beginning = torch.nonzero(beginning)

            global_attention_mask = torch.zeros(decoder_input.shape, dtype=torch.long, device=decoder_input.device)
            global_attention_mask[beginning,[0]] = 1

            outputs = model(input_ids=encoder_input, attention_mask=encoder_attention_mask, decoder_input_ids=decoder_input, decoder_attention_mask=decoder_attention_mask, global_attention_mask=global_attention_mask).logits  # Generate predictions
            
            # Free GPU memory
            del encoder_input
            del encoder_attention_mask
            del decoder_input
            del decoder_attention_mask
            del global_attention_mask

            labels = labels.to(device)
            
            loss = criterion(outputs.view(-1, model.config.vocab_size), labels.view(-1))
            total_val_loss += loss.item() * labels.size(0)  # Accumulate the loss

            preds = outputs.argmax(dim=-1)  # Get predicted classes
            correct = (preds == labels).sum().item()
            accuracy = correct / labels.numel()
            total_accuracy += accuracy * labels.size(0)  # Accumulate the accuracy

            # Collect predictions and labels for calculating additional metrics
            preds_all.extend(preds.view(-1).tolist())
            labels_all.extend(labels.view(-1).tolist())
            # Update the progress bar
            if epoch is None:
                val_loader.set_description("Testing")
            else:
                val_loader.set_description(f"Epoch {epoch} - Validation")
            val_loader.update(1)

    # Calculate overall metrics from collected predictions and labels
    avg_val_loss = total_val_loss / val_dataset_size
    avg_val_accuracy = total_accuracy / val_dataset_size
    precision = precision_score(labels_all, preds_all, average='macro', zero_division=0)
    recall = recall_score(labels_all, preds_all, average='macro', zero_division=0)
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    kappa = cohen_kappa_score(labels_all, preds_all)

    return avg_val_loss, avg_val_accuracy, precision, recall, f1, kappa

genomes = load_dataset(input_file)
unique_tokens = set(token for genome in genomes for token in genome.split())
actual_vocab_size = len(unique_tokens)
vocab_size = min(actual_vocab_size, max_vocab_size)
num_sequences = len(genomes)
sequence_lengths = [len(genome.split()) for genome in genomes]
min_sequence_length = min(sequence_lengths)
max_sequence_length = max(sequence_lengths)
avg_sequence_length = sum(sequence_lengths) / num_sequences

logging.info(
    f"Dataset loaded: {num_sequences} sequences\n"
    f"Sequence lengths - Min: {min_sequence_length}, Max: {max_sequence_length}, Avg: {avg_sequence_length:.2f}"
)


#tokenizer = ByteLevelBPETokenizer()
#tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit(" ")
#tokenizer.train_from_iterator(genomes, vocab_size=vocab_size, special_tokens=["<s>","<pad>", "</s>","<unk>", "<mask>",])
#Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)
#tokenizer.save_model(tokenizer_dir)
tokenizer = LEDTokenizer.from_pretrained(tokenizer_dir, add_prefix_space=True)
vocab_size = tokenizer.vocab_size

if train_size + val_size > 1.0:
    raise ValueError("The sum of train_size and val_size must be less than or equal to 1.0")
if train_size + val_size == 1.0:
    train_genomes, val_genomes = train_test_split(genomes, train_size=train_size, random_state=seed)
    test_genomes = []
else:
    train_genomes, temp_genomes = train_test_split(genomes, train_size=train_size, random_state=seed)
    val_genomes, test_genomes = train_test_split(temp_genomes, test_size=1.0 - val_size / (1.0 - train_size), random_state=seed)

class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer input.

    This module adds positional encoding to the input embeddings to provide positional information
    to the transformer model.

    Args:
    - d_model (int): Dimension of the input embeddings.
    - max_len (int): Maximum length of the input sequence.
    - dropout (float): Dropout probability.

    Attributes:
    - pe (torch.Tensor): Positional encoding tensor.

    Methods:
    - forward(x): Forward pass of the positional encoding module.
    """

    def __init__(self, d_model, max_len, dropout=pe_dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

# class SimpleTransformerModel(nn.Module):
#     """
#     Simple transformer model for genomic data.

#     This class defines a simple transformer model architecture for processing genomic data.

#     Args:
#     - vocab_size (int): Size of the vocabulary.
#     - embed_dim (int): Dimension of the input embeddings.
#     - num_heads (int): Number of attention heads.
#     - num_layers (int): Number of transformer layers.
#     - max_seq_length (int): Maximum sequence length.
#     - dropout_rate (float): Dropout rate.
#     - pe_max_len (int): Maximum length for positional encoding.
#     - pe_dropout_rate (float): Dropout rate for positional encoding.

#     Methods:
#     - forward(x): Forward pass of the transformer model.

#     Reference:
#     - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., 
#     Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I., 2017. 
#     Attention is all you need. Advances in neural information processing systems, 30.
#     """

#     def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length, dropout_rate, pe_max_len, pe_dropout_rate):
#         super(SimpleTransformerModel, self).__init__()
#         self.pos_encoding = PositionalEncoding(embed_dim, pe_max_len, dropout=pe_dropout_rate)
#         self.vocab_size = vocab_size
#         self.embed = nn.Embedding(vocab_size, embed_dim)
#         transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
#         self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
#         self.out = nn.Linear(embed_dim, vocab_size)

#     def forward(self, x):
#         x = self.embed(x)
#         x = self.pos_encoding(x)  # Apply positional encoding after embedding
#         x = self.transformer(x)
#         return self.out(x)

# class LongformerModel(nn.Module):
#     """
#     Longformer model for processing long sequences.

#     This class defines a Longformer model that can handle long input sequences efficiently
#     by using a combination of local and global attention mechanisms. It is based on the
#     Longformer architecture introduced by Beltagy et al. (2020).

#     Args:
#     - vocab_size (int): Size of the vocabulary.
#     - embed_dim (int): Dimension of the input embeddings.
#     - num_heads (int): Number of attention heads.
#     - num_layers (int): Number of Longformer layers.
#     - max_seq_length (int): Maximum sequence length.
#     - dropout_rate (float): Dropout rate for the model.
#     - pe_max_len (int): Maximum length for positional encoding.
#     - pe_dropout_rate (float): Dropout rate for positional encoding.
#     - longformer_config (LongformerConfig): Configuration object for the Longformer model.

#     Attributes:
#     - pos_encoding (PositionalEncoding): Positional encoding module.
#     - vocab_size (int): Size of the vocabulary.
#     - embed (nn.Embedding): Embedding layer for input tokens.
#     - longformer_layers (nn.ModuleList): List of Longformer self-attention layers.
#     - out (nn.Linear): Output linear layer for token prediction.

#     Methods:
#         forward(x): Perform forward pass through the Longformer model.

#     References:
#         - Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer.
#           arXiv preprint arXiv:2004.05150.
#     """

#     def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length,
#                  dropout_rate, pe_max_len, pe_dropout_rate, longformer_config):
#         super(LongformerModel, self).__init__()
#         self.pos_encoding = PositionalEncoding(embed_dim, pe_max_len, dropout=pe_dropout_rate)
#         self.vocab_size = vocab_size
#         self.embed = nn.Embedding(vocab_size, embed_dim)

#         self.longformer_layers = nn.ModuleList([
#             LongformerSelfAttention(longformer_config, layer_id=i)
#             for i in range(num_layers)
#         ])

#         self.out = nn.Linear(embed_dim, vocab_size)

#     def forward(self, x):
#         x = self.embed(x)
#         x = self.pos_encoding(x)

#         attention_mask = torch.ones(x.size()[:-1], dtype=torch.long, device=x.device)

#         # Generate is_index_masked tensor
#         is_index_masked = torch.zeros_like(attention_mask, dtype=torch.bool)

#         for longformer_layer in self.longformer_layers:
#             x = longformer_layer(x, attention_mask=attention_mask, is_index_masked=is_index_masked)[0]

#         return self.out(x)

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
        self.mask_token = self.tokenizer.mask_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        input = self.tokenizer.encode(text)

        beginning = 0

        # ensure you don't attend to <mask>, set 0 in attention_mask
        # Ensure the sequence is not longer than max_length, take random slice
        if len(input) >= self.max_length:
            # start at random point in sequence
            start_index = random.randint(0, len(input) - self.max_length)

            labels = input[start_index:start_index + self.max_length]
            # wrap decoder input to right
            decoder_input = [labels[-1]]
            decoder_input.extend(labels[:-1])
            
            # decode to get input string, then mask and re-encode to ensure same string is learned from in decoder and encoder
            # mask will remove characters, so indexes do not map between decoder_input and encoder_input
            text = tokenizer.decode(labels, skip_special_tokens=True)
            #print("pre-masking")
            #print(text)
            text_masked = mask_integers(text, self.prop_masked)
            #print("after masking")
            #print(text_masked)

            # encode, removing <s> and </s> token if not at end of genome
            if start_index == (len(input) - self.max_length - 1):
                encoder_input = self.tokenizer.encode(text_masked)
            elif start_index == 0:
                encoder_input = self.tokenizer.encode(text_masked)[:-1]
            else:
                encoder_input = self.tokenizer.encode(text_masked)[1:-1]

            beginning = 1 if start_index == 0 else 0
        else:
            # generate decoder and labels input, wrapping decoder input to right
            labels = input
            decoder_input = [labels[-1]]
            decoder_input.extend(labels[:-1])
            
            text_masked = mask_integers(text, self.prop_masked)
            encoder_input = self.tokenizer.encode(text_masked)
            beginning = 1

        len_decoder = len(decoder_input)
        decoder_input = pad_input(decoder_input, self.max_length, tokenizer)

        decoder_attention_mask = torch.ones(len(decoder_input), dtype=torch.long)
        decoder_attention_mask[len_decoder:] = 0

        labels = pad_input(labels, self.max_length, tokenizer, labels=True)

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
        encoder_input = pad_input(encoder_input, self.max_length, tokenizer)

        #print('encoder_input post padding')
        #print(encoder_input)

        # do not attend to mask tokens
        #print(int(self.mask_token))
        #mask_idx = np.flatnonzero(np.array(encoder_input) == int(self.mask_token))

        encoder_attention_mask = torch.ones(len(encoder_input), dtype=torch.long)
        encoder_attention_mask[len_masked:] = 0
        #encoder_attention_mask[mask_idx] = 0

        #print("labels")
        #print(labels)
        #print("decoder_input")
        #print(decoder_input)
        #print("decoder_attention_mask")
        #print(decoder_attention_mask.tolist())
        #print("encoder_input")
        #print(encoder_input)
        #print("encoder_attention_mask")
        #print(encoder_attention_mask.tolist())

        return torch.tensor(decoder_input, dtype=torch.long), torch.tensor(encoder_input, dtype=torch.long), torch.tensor(labels, dtype=torch.long), decoder_attention_mask, encoder_attention_mask, beginning

class EarlyStopping:
    """
    Early stopping handler for model training.

    This class implements early stopping functionality to stop model training
    when the validation loss stops improving.

    Args:
    - patience (int): Number of epochs to wait before stopping.
    - min_delta (float): Minimum change in loss to be considered an improvement.
    - verbose (bool): Whether to print early stopping messages.

    Methods:
    - __call__(val_loss): Check if early stopping criteria are met based on the validation loss.
    """

    def __init__(self, patience, min_delta=min_delta, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# if model_type == 'transformer':
#     model = SimpleTransformerModel(vocab_size, embed_dim, num_heads, num_layers, max_seq_length, dropout_rate=model_dropout_rate, pe_max_len=pe_max_len, pe_dropout_rate=pe_dropout_rate)
# elif model_type == 'longformer':
#     attention_window = args.attention_window
#     longformer_config = LongformerConfig(
#         hidden_size=embed_dim,
#         num_attention_heads=num_heads,
#         num_hidden_layers=num_layers,
#         attention_window=[attention_window] * num_layers,
#         intermediate_size=4 * embed_dim,
#     )
#     model = LongformerModel(vocab_size, embed_dim, num_heads, num_layers, max_seq_length, dropout_rate=model_dropout_rate, pe_max_len=pe_max_len, pe_dropout_rate=pe_dropout_rate, longformer_config=longformer_config)
#elif model_type == 'BARTlongformer':
    # from https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/led
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
    attention_window = args.attention_window
)
model = LEDForConditionalGeneration(BARTlongformer_config)
# else:
#     raise ValueError(f"Invalid model type: {model_type}")

early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=min_delta, verbose=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}", flush=True)

# training dataset
train_dataset = GenomeDataset(train_genomes, tokenizer, max_seq_length, prop_masked)
#if args.model_type == "longformer" or args.model_type == "BARTlongformer":
train_dataset.attention_window = attention_window
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_dataset_size = len(train_loader.dataset)

# validation dataset
val_dataset = GenomeDataset(val_genomes, tokenizer, max_seq_length, prop_masked)
#if args.model_type == "longformer" or args.model_type == "BARTlongformer":
val_dataset.attention_window = attention_window
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_dataset_size = len(val_loader.dataset)

criterion = torch.nn.CrossEntropyLoss() # what are we trying to optimize?
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # How are we trying to optimizer it?
lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=lr_scheduler_factor, patience=lr_patience) # taking big, then small steps

if torch.cuda.is_available():
    print("GPU available, using cuda, device: {}".format(str(args.device)))

    device = torch.device("cuda:" + str(args.device)) # Run on a GPU if one is available
else:
    print("GPU not available, using cpu.")
    device = torch.device("cpu")
logging.info(f"device = {device}")
model.to(device)

start_epoch, is_checkpoint_loaded = load_checkpoint(model, optimizer, model_save_path, restart)

if is_checkpoint_loaded:
    logging.info("Continuing training from the loaded checkpoint.")
else:
    logging.info("Starting training from scratch.")
    start_epoch = 0

print(f"vocab_size: {vocab_size} | embed_dim: {embed_dim} | num_heads: {num_heads} | num_layers: {num_layers} | max_seq_length: {max_seq_length}", flush=True)
for epoch in range(start_epoch, epochs):
    writer = SummaryWriter(log_dir=log_dir)
    # Training model loop
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch} - Training", unit="batch")
    avg_train_loss = train_model(train_loader, model, optimizer, criterion, device)
    train_perplexity = torch.exp(torch.tensor(avg_train_loss))
    # Log training metrics
    logging.info(f'Epoch {epoch} - Training Loss: {avg_train_loss}, Perplexity: {train_perplexity}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Perplexity/train", train_perplexity, epoch)

    # Validate model loop
    val_loader = tqdm(val_loader, desc=f"Epoch {epoch} - Validation", unit="batch")
    avg_val_loss, val_accuracy, val_precision, val_recall, val_f1, val_kappa = validate_model(val_loader, model, criterion, device, epoch)
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))
    # Log validation metrics
    logging.info(f'Epoch {epoch} - Validation Loss: {avg_val_loss}, Perplexity: {val_perplexity}, Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}, Kappa: {val_kappa}')
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Perplexity/val", val_perplexity, epoch)
    writer.add_scalar("Accuracy/val", val_accuracy, epoch)
    writer.add_scalar("Precision/val", val_precision, epoch)
    writer.add_scalar("Recall/val", val_recall, epoch)
    writer.add_scalar("F1/val", val_f1, epoch)
    writer.add_scalar("Kappa/val", val_kappa, epoch)
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

    lr_scheduler.step(avg_val_loss)
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered.", flush=True)
        break
    elif avg_val_loss <= early_stopping.best_loss:
        print("Saving model checkpoint.", flush=True)
        save_checkpoint(model, optimizer, epoch, avg_train_loss, model_save_path)

    gc.collect()
    writer.close()

if len(test_genomes) > 0:
    test_dataset = GenomeDataset(test_genomes, tokenizer, max_seq_length, prop_masked)
    #if args.model_type == "longformer" or args.model_type == "BARTlongformer":
    test_dataset.attention_window = attention_window
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_dataset_size = len(test_loader.dataset)  # Store the size of the test dataset
    test_loader = tqdm(test_loader, desc="Testing", unit="batch")
    # Test Model Loop
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_kappa = validate_model(test_loader, model, criterion, device)
    test_perplexity = torch.exp(torch.tensor(test_loss))
    # Log test metrics
    logging.info(f'Test Loss: {test_loss}, Perplexity: {test_perplexity}, Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}, Kappa: {test_kappa}')
    # Create a new SummaryWriter instance for test metrics
    test_writer = SummaryWriter(log_dir=os.path.join(log_dir, "test"))
    test_writer.add_scalar("Loss/test", test_loss)
    test_writer.add_scalar("Perplexity/test", test_perplexity)
    test_writer.add_scalar("Accuracy/test", test_accuracy)
    test_writer.add_scalar("Precision/test", test_precision)
    test_writer.add_scalar("Recall/test", test_recall)
    test_writer.add_scalar("F1/test", test_f1)
    test_writer.add_scalar("Kappa/test", test_kappa)
    test_writer.close()

else:
    print("No test set available for evaluation.", flush=True)


