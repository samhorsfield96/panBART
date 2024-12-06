import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
import psutil
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import warnings
import random
import numpy as np
from tqdm import tqdm
from transformers import LEDConfig, LEDForConditionalGeneration, LEDTokenizer
from pathlib import Path
import random
import re

# Global variables
PROGRAM_NAME = "panBART"
VERSION = "0.1.0"
AUTHOR = "Samuel Horsfield"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("training.log"), 
        logging.StreamHandler()
    ],
)

# DDP setup
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

def pad_input(input, max_length, pad_token_id, labels=False):

    len_masked = len(input)
    if labels == False:
        input.extend([pad_token_id] * (max_length - len_masked))
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

    
    args = parser.parse_args()

    # Ensure max_seq_length is greater than or equal to attention_window
    args.max_seq_length = max(args.max_seq_length, args.attention_window)
    # Round down max_seq_length to the nearest multiple of attention_window
    args.max_seq_length = (args.max_seq_length // args.attention_window) * args.attention_window


    return args

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
            # randomise conitg order
            #for index in range(len(genomes)):
            #    split_genome = genomes[index].split(" _ ")
            #    random.shuffle(split_genome)
            #    genome = "_ " + " _ ".join(split_genome) + " _"
            #    genomes[index] = genome
        return genomes
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit(1)

def save_checkpoint(model, optimizer, epoch, loss, lr_scheduler, save_path):
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
            "lr_scheduler" : lr_scheduler,
            "loss": loss,
        }
        torch.save(checkpoint, save_path)
    except IOError as e:
        print(f"Failed to save checkpoint to '{save_path}': {e}")

def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, restart, rank, map_location=None):
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
        if map_location != None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler = checkpoint["lr_scheduler"]
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

def train_model(train_loader, model, optimizer, criterion, device, vocab_size, encoder_only, epoch):
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
    for i, (decoder_input, encoder_input, labels, decoder_attention_mask, encoder_attention_mask, global_attention_mask) in enumerate(train_loader):  # Added enumeration for clarity
        
        optimizer.zero_grad()  # Clear gradients before calculating them
        if encoder_only:
            encoder_input, encoder_attention_mask, global_attention_mask = encoder_input.to(device), encoder_attention_mask.to(device), global_attention_mask.to(device)  # Move data to the appropriate device
            
            outputs = model(input_ids=encoder_input, attention_mask=encoder_attention_mask, global_attention_mask=global_attention_mask).logits  # Generate predictions
        else:
            decoder_input, encoder_input, decoder_attention_mask, encoder_attention_mask, global_attention_mask = decoder_input.to(device), encoder_input.to(device), decoder_attention_mask.to(device), encoder_attention_mask.to(device), global_attention_mask.to(device)  # Move data to the appropriate device
            
            outputs = model(input_ids=encoder_input, attention_mask=encoder_attention_mask, decoder_input_ids=decoder_input, decoder_attention_mask=decoder_attention_mask, global_attention_mask=global_attention_mask).logits  # Generate predictions

        # Free GPU memory
        del encoder_input
        del encoder_attention_mask
        del decoder_input
        del decoder_attention_mask
        del global_attention_mask

        #torch.cuda.empty_cache()
        
        labels = labels.to(device)
        
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

        total_train_loss += loss.item() * labels.size(0)  # Accumulate the loss
        
        del labels
        #torch.cuda.empty_cache()

        loss.backward()  # Compute gradient of the loss w.r.t. network parameters
        optimizer.step()  # Update parameters based on gradient
        
        # Update the progress bar
        train_loader.set_description(f"Epoch {epoch} - Training")
        train_loader.update(1)

    #avg_train_loss = total_train_loss / train_dataset_size
    return total_train_loss

def validate_model(val_loader, model, criterion, device, vocab_size, encoder_only, epoch=None):
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
        for decoder_input, encoder_input, labels, decoder_attention_mask, encoder_attention_mask, global_attention_mask in val_loader:  # Correctly unpack the tuples returned by the DataLoader
            if encoder_only:
                encoder_input, encoder_attention_mask, global_attention_mask = encoder_input.to(device), encoder_attention_mask.to(device), global_attention_mask.to(device)  # Move data to the appropriate device
                
                outputs = model(input_ids=encoder_input, attention_mask=encoder_attention_mask, global_attention_mask=global_attention_mask).logits  # Generate predictions
            else:
                decoder_input, encoder_input, decoder_attention_mask, encoder_attention_mask, global_attention_mask = decoder_input.to(device), encoder_input.to(device), decoder_attention_mask.to(device), encoder_attention_mask.to(device), global_attention_mask.to(device)  # Move data to the appropriate device
                
                outputs = model(input_ids=encoder_input, attention_mask=encoder_attention_mask, decoder_input_ids=decoder_input, decoder_attention_mask=decoder_attention_mask, global_attention_mask=global_attention_mask).logits  # Generate predictions

            # Free GPU memory
            del encoder_input
            del encoder_attention_mask
            del decoder_input
            del decoder_attention_mask
            del global_attention_mask

            #torch.cuda.empty_cache()

            labels = labels.to(device)
            
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            total_val_loss += loss.item() * labels.size(0)  # Accumulate the loss

            preds = outputs.argmax(dim=-1)  # Get predicted classes

            # ignore padding positions
            mask = labels != -100
            preds = preds[mask]
            labels = labels[mask]

            # calculate accuracy
            correct = (preds == labels).sum().item()
            accuracy = correct / labels.numel()
            total_accuracy += accuracy * labels.size(0)  # Accumulate the accuracy

            # print("loss:")
            # print(loss.item())
            # print("preds:")
            # print(preds.tolist())
            # print("labels:")
            # print(labels.tolist())

            #print("preds masked:")
            #print(preds.tolist())
            #print("label masked:")
            #print(labels.tolist())

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
    #avg_val_loss = total_val_loss / dataset_size
    #avg_val_accuracy = total_accuracy / dataset_size
    precision = precision_score(labels_all, preds_all, average='macro', zero_division=0)
    recall = recall_score(labels_all, preds_all, average='macro', zero_division=0)
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    kappa = cohen_kappa_score(labels_all, preds_all)

    return total_val_loss, total_accuracy, precision, recall, f1, kappa

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

        # randomise contig order and flip randomly
        split_genome = text.split("_")
        flip_contigs = [random.random() < 0.5 for _ in range(len(split_genome))]
        #print(split_genome)
        #print(flip_contigs)

        for index, contig in enumerate(split_genome):
            if flip_contigs[index]:
                split_genome[index] = " ".join([str(int(x) * -1) for x in contig.strip().split()][::-1])
            else:
                split_genome[index] = contig.strip()

        #print(split_genome)
        random.shuffle(split_genome)
        genome = "_ " + " _ ".join(split_genome) + " _"

        input = self.tokenizer.encode(genome).ids

        beginning = 0

        # Ensure the sequence is not longer than max_length, take random slice
        if len(input) >= self.max_length:
            # start at random point in sequence
            start_index = random.randint(1, len(input) - self.max_length)

            labels = input[start_index:start_index + self.max_length]
            # wrap decoder input to right
            decoder_input = input[start_index - 1:(start_index + self.max_length) - 1]
            
            # decode to get input string, then mask and re-encode to ensure same string is learned from in decoder and encoder
            # mask will remove characters, so indexes do not map between decoder_input and encoder_input
            text = self.tokenizer.decode(labels, skip_special_tokens=False)
            # print("pre-masking")
            # print(text)
            text_masked = mask_integers(text, self.prop_masked)
            # print("after masking")
            # print(text_masked)

            encoder_input = self.tokenizer.encode(text_masked).ids

            beginning = 1 if start_index == 1 else 0
        else:
            # generate decoder and labels input, wrapping decoder input to right
            labels = input[1:]

            # mask original text
            text = self.tokenizer.decode(labels, skip_special_tokens=False)
            text_masked = mask_integers(text, self.prop_masked)

            encoder_input = self.tokenizer.encode(text_masked).ids
            decoder_input = input[:-1]
            beginning = 1

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

    def __init__(self, patience, min_delta, verbose=False):
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

def run_model(rank, world_size, args, early_stopping, BARTlongformer_config, train_genomes, val_genomes, test_genomes, tokenizer, vocab_size, DDP_active=False):
    if DDP_active:
        setup(rank, world_size)

    attention_window=args.attention_window
    max_seq_length = args.max_seq_length
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    lr_scheduler_factor = args.lr_scheduler_factor
    weight_decay = args.weight_decay
    lr_patience = args.lr_patience
    epochs = args.epochs
    model_save_path = args.model_save_path
    log_dir = args.log_dir
    prop_masked = args.prop_masked
    restart = args.restart
    num_workers = args.num_workers

    # determine number of GPUs to use
    #num_gpus = torch.cuda.device_count()
    model = LEDForConditionalGeneration(BARTlongformer_config)
    if args.gradient_checkpointing == True and DDP_active == False:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    
    # print model params
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total model parameters: {pytorch_total_params}")
    
    device = rank
    model = model.to(device)
    logging.info(f"device = {device}")
    if DDP_active:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        train_sampler = DistributedSampler(train_genomes, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_genomes, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_genomes, num_replicas=world_size, rank=rank)
        num_workers = 0
        pin_memory = False
        shuffle = False
    else:
        train_sampler, val_sampler, test_sampler = None, None, None
        pin_memory = True
        shuffle = True
    
    # training dataset
    train_dataset = GenomeDataset(train_genomes, tokenizer, max_seq_length, prop_masked)
    train_dataset.attention_window = attention_window
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler)
    train_dataset_size = len(train_loader.dataset)

    # validation dataset
    
    val_dataset = GenomeDataset(val_genomes, tokenizer, max_seq_length, prop_masked)
    val_dataset.attention_window = attention_window
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, sampler=val_sampler)
    val_dataset_size = len(val_loader.dataset)

    criterion = torch.nn.CrossEntropyLoss().to(device) # what are we trying to optimize?
    # scale lr by number of GPUs used https://github.com/Lightning-AI/pytorch-lightning/discussions/3706#discussioncomment-3960433
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate * math.sqrt(world_size), weight_decay=weight_decay) # How are we trying to optimizer it?
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=lr_scheduler_factor, patience=lr_patience) # taking big, then small steps

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    map_location = None
    if DDP_active:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        dist.barrier()
    start_epoch, is_checkpoint_loaded = load_checkpoint(model, optimizer, lr_scheduler, model_save_path, restart, map_location)

    if rank == 0:
        if is_checkpoint_loaded:
            logging.info("Continuing training from the loaded checkpoint.")
        else:
            logging.info("Starting training from scratch.")

    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, epochs):
        # Training model loop
        #train_loader.sampler.set_epoch(epoch)
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch} - Training", unit="batch")
        total_train_loss = train_model(train_loader, model, optimizer, criterion, device, vocab_size, args.encoder_only, epoch)

        total_train_loss_tensor = torch.tensor(total_train_loss).to(rank)

        if DDP_active:
            dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = total_train_loss_tensor.item() / train_dataset_size
        train_perplexity = torch.exp(torch.tensor(avg_train_loss))
        
        # Log training metrics
        if (DDP_active and rank == 0) or DDP_active == False:  # Only rank 0 should write logs
            logging.info(f'Epoch {epoch} - Training Loss: {avg_train_loss}, Perplexity: {train_perplexity}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Perplexity/train", train_perplexity, epoch)
            writer.add_scalar("Learning_rate/train", optimizer.param_groups[0]["lr"], epoch)

        # Validate model loop
        #val_loader.sampler.set_epoch(epoch)
        val_loader = tqdm(val_loader, desc=f"Epoch {epoch} - Validation", unit="batch")
        total_val_loss, total_accuracy, val_precision, val_recall, val_f1, val_kappa = validate_model(val_loader, model, criterion, device, vocab_size, args.encoder_only, epoch)
        
        total_val_loss_tensor = torch.tensor(total_val_loss).to(rank)
        total_accuracy_tensor = torch.tensor(total_accuracy).to(rank)
        val_precision_tensor = torch.tensor(val_precision).to(rank)
        val_recall_tensor = torch.tensor(val_recall).to(rank)
        val_f1_tensor = torch.tensor(val_f1).to(rank)
        val_kappa_tensor = torch.tensor(val_kappa).to(rank)

        # get results from all GPUs
        if DDP_active:
            dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_accuracy_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_precision_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_recall_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_f1_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_kappa_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = total_val_loss_tensor.item() / val_dataset_size
        val_perplexity = torch.exp(torch.tensor(avg_val_loss))
        val_accuracy = total_accuracy_tensor.item() / val_dataset_size
        val_precision = val_precision_tensor.item() / world_size
        val_recall = val_recall_tensor.item() / world_size
        val_f1 = val_f1_tensor.item() / world_size
        val_kappa = val_kappa_tensor.item() / world_size

        # step lr_scheduler, will be synchronised as same value computed across GPUs
        lr_scheduler.step(avg_val_loss)

        # Log validation metrics
        if (DDP_active and rank == 0) or DDP_active == False:  # Only rank 0 should write logs
            logging.info(f'Epoch {epoch} - Validation Loss: {avg_val_loss}, Perplexity: {val_perplexity}, Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}, Kappa: {val_kappa}')
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Perplexity/val", val_perplexity, epoch)
            writer.add_scalar("Accuracy/val", val_accuracy, epoch)
            writer.add_scalar("Precision/val", val_precision, epoch)
            writer.add_scalar("Recall/val", val_recall, epoch)
            writer.add_scalar("F1/val", val_f1, epoch)
            writer.add_scalar("Kappa/val", val_kappa, epoch)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

            early_stopping(avg_val_loss)

            early_stop_tensor = torch.tensor(int(early_stopping.early_stop)).to(rank)
            
            if avg_val_loss <= early_stopping.best_loss:
                print("Saving model checkpoint.", flush=True)
                save_checkpoint(model, optimizer, epoch, avg_train_loss, lr_scheduler, model_save_path)

            gc.collect()
            writer.close()
        elif (DDP_active and rank != 0):
            early_stop_tensor = torch.tensor(0).to(rank)

        # broadcast to all GPUs, check if early stop triggered
        if DDP_active:
            dist.broadcast(early_stop_tensor, src=0)
        if early_stop_tensor.item() == 1:
            print("Early stopping triggered.", flush=True)
            break

    if len(test_genomes) > 0:
        test_dataset = GenomeDataset(test_genomes, tokenizer, max_seq_length, prop_masked)
        test_dataset.attention_window = attention_window
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, sampler=test_sampler)
        #test_loader.sampler.set_epoch(epoch)
        test_dataset_size = len(test_loader.dataset)  # Store the size of the test dataset
        test_loader = tqdm(test_loader, desc="Testing", unit="batch")
        # Test Model Loop
        total_test_loss, total_test_accuracy, test_precision, test_recall, test_f1, test_kappa = validate_model(test_loader, model, criterion, device, vocab_size, args.encoder_only)
        
        total_test_loss_tensor = torch.tensor(total_test_loss).to(rank)
        total_accuracy_tensor = torch.tensor(total_test_accuracy).to(rank)
        test_precision_tensor = torch.tensor(test_precision).to(rank)
        test_recall_tensor = torch.tensor(test_recall).to(rank)
        test_f1_tensor = torch.tensor(test_f1).to(rank)
        test_kappa_tensor = torch.tensor(test_kappa).to(rank)

        # get results from all GPUs
        if DDP_active:
            dist.all_reduce(total_test_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_accuracy_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_precision_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_recall_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_f1_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_kappa_tensor, op=dist.ReduceOp.SUM)
        
        avg_test_loss = total_test_loss_tensor.item() / test_dataset_size
        test_perplexity = torch.exp(torch.tensor(avg_test_loss))
        test_accuracy = total_accuracy_tensor.item() / test_dataset_size
        test_precision = test_precision_tensor.item() / world_size
        test_recall = test_recall_tensor.item() / world_size
        test_f1 = test_f1_tensor.item() / world_size
        test_kappa = test_kappa_tensor.item() / world_size

        # Log test metrics
        if (DDP_active and rank == 0) or DDP_active == False:  # Only rank 0 should write logs
            logging.info(f'Test Loss: {avg_test_loss}, Perplexity: {test_perplexity}, Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}, Kappa: {test_kappa}')
            # Create a new SummaryWriter instance for test metrics
            test_writer = SummaryWriter(log_dir=os.path.join(log_dir, "test"))
            test_writer.add_scalar("Loss/test", avg_test_loss)
            test_writer.add_scalar("Perplexity/test", test_perplexity)
            test_writer.add_scalar("Accuracy/test", test_accuracy)
            test_writer.add_scalar("Precision/test", test_precision)
            test_writer.add_scalar("Recall/test", test_recall)
            test_writer.add_scalar("F1/test", test_f1)
            test_writer.add_scalar("Kappa/test", test_kappa)
            test_writer.close()

    else:
        print("No test set available for evaluation.", flush=True)
    
    if DDP_active:
        cleanup()
    
def main():
    args = parse_args()

    params = vars(args)  # Convert the parsed arguments to a dictionary

    input_file = args.input_file
    train_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file
    device = args.device
    attention_window=args.attention_window
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    num_layers = args.num_layers
    max_seq_length = args.max_seq_length
    model_dropout_rate = args.model_dropout_rate
    early_stop_patience =args.early_stop_patience
    min_delta = args.min_delta
    max_vocab_size = args.max_vocab_size
    model_save_path = args.model_save_path
    tokenizer_path = args.tokenizer_path
    train_size = args.train_size
    val_size = args.val_size
    seed = args.seed
    max_input_len = args.max_input_len
    min_input_len = args.min_input_len

    # Check if max_seq_length is a multiple of attention_window when using Longformer
    #if (model_type == "longformer" or model_type == "BARTlongformer") and max_seq_length % attention_window != 0:
    if max_seq_length % attention_window != 0:
        logging.info(f"Error: When using the LED model, the maximum sequence length (max_seq_length) must be a multiple of the attention window size (attention_window).")
        logging.info(f"Current values: max_seq_length = {max_seq_length}, attention_window = {attention_window}")
        logging.info("Please adjust these values and try again.")
        exit(1)

    if input_file != None:
        # Check whether certain files exist
        if not os.path.isfile(input_file):
            print(f"Error: The specified input file '{input_file}' does not exist.")
            exit(1)
        genomes = load_dataset(input_file)
            # remove sequences that are too long or short
        if max_input_len != None:
            genomes = [genome for genome in genomes if len(genome.split()) <= max_input_len]

        if min_input_len != None:
            genomes = [genome for genome in genomes if len(genome.split()) >= min_input_len]
    else:
        # read in pre-split genomes
        train_genomes = load_dataset(train_file)
        val_genomes = load_dataset(val_file)
        test_genomes = load_checkpoint(test_file)
        if max_input_len != None:
            train_genomes = [genome for genome in train_genomes if len(genome.split()) <= max_input_len]
            val_genomes = [genome for genome in val_genomes if len(genome.split()) <= max_input_len]
            test_genomes = [genome for genome in test_genomes if len(genome.split()) <= max_input_len]

        if min_input_len != None:
            train_genomes = [genome for genome in train_genomes if len(genome.split()) >= min_input_len]
            val_genomes = [genome for genome in val_genomes if len(genome.split()) >= min_input_len]
            test_genomes = [genome for genome in test_genomes if len(genome.split()) >= min_input_len]

        # combine for tokenizer
        genomes = train_genomes + val_genomes + test_genomes


    if model_save_path and not os.path.isdir(os.path.dirname(model_save_path)):
        print(f"Error: The directory for model save path '{model_save_path}' does not exist.")
        exit(1)
    
    # generate tokenizer information
    set_seed(args.seed)
    print_parameters_table(params)
    unique_tokens = set(token for genome in genomes for token in genome.split())
    actual_vocab_size = int(len(unique_tokens))
    if max_vocab_size is not None:
        vocab_size = min(actual_vocab_size, max_vocab_size)
    else:
        vocab_size = actual_vocab_size
    
    sequence_lengths = [len(genome.split()) for genome in genomes]
    num_sequences = len(genomes)
    min_sequence_length = min(sequence_lengths)
    max_sequence_length = max(sequence_lengths)
    avg_sequence_length = sum(sequence_lengths) / num_sequences

    logging.info(
        f"Dataset loaded: {num_sequences} sequences\n"
        f"Sequence lengths - Min: {min_sequence_length}, Max: {max_sequence_length}, Avg: {avg_sequence_length:.2f}"
    )

    if not args.reuse_tokenizer:
        tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit(" ")
        trainer = trainers.WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"], vocab_size=vocab_size)
        tokenizer.train_from_iterator(genomes, trainer)
        tokenizer.save(tokenizer_path)

    print_banner()

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
        max_encoder_position_embeddings=max_seq_length,
        max_decoder_position_embeddings=max_seq_length,
        dropout=model_dropout_rate,
        attention_window = args.attention_window,
        pad_token_id=tokenizer.encode("<pad>").ids[0],
        bos_token_id=tokenizer.encode("<s>").ids[0],
        eos_token_id=tokenizer.encode("</s>").ids[0]
        )

    early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=min_delta, verbose=True)
    #total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Total number of trainable parameters: {total_params}", flush=True)

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

    # split genomes if required
    if input_file != None
        if train_size + val_size > 1.0:
            raise ValueError("The sum of train_size and val_size must be less than or equal to 1.0")
        if train_size + val_size == 1.0:
            train_genomes, val_genomes = train_test_split(genomes, train_size=train_size, random_state=seed)
            test_genomes = []
        else:
            train_genomes, temp_genomes = train_test_split(genomes, train_size=train_size, random_state=seed)
            val_genomes, test_genomes = train_test_split(temp_genomes, test_size=1.0 - val_size / (1.0 - train_size), random_state=seed)

    # delete genomes list from memory
    del genomes

    # print test genomes to file
    if len(test_genomes) > 0 and args.save_test_data:
        file_base, _ = os.path.splitext(model_save_path)
        save_path = file_base + "_test_genomes.txt"
        with open(save_path, "w") as o:
            for entry in test_genomes:
                o.write(entry + "\n")

    print(f"vocab_size: {vocab_size} | embed_dim: {embed_dim} | num_heads: {num_heads} | num_layers: {num_layers} | max_seq_length: {max_seq_length}", flush=True)
    if DDP_active:
        mp.spawn(run_model,
                args=(world_size, args, early_stopping, BARTlongformer_config, train_genomes, val_genomes, test_genomes, tokenizer, vocab_size, DDP_active),
                nprocs=world_size,
                join=True)
    else:
        run_model(device, 1, args, early_stopping, BARTlongformer_config, train_genomes, val_genomes, test_genomes, tokenizer, vocab_size)

if __name__ == "__main__":
    main()