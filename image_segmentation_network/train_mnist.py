from __future__ import absolute_import, division, print_function
# These imports ensure compatibility between Python 2 and Python 3 for important language features.

import argparse  # For parsing command-line arguments
import os  # For interacting with the file system
import pprint  # Pretty-print for better formatted logging

import torch  # Main PyTorch library
import torch.optim as optim  # Optimizers in PyTorch
import numpy as np  # For numerical operations
import random  # For controlling randomness

# Importing training, validation, loss functions, and utilities from the project
from lib.core.function import train  # Function to train the model
from lib.core.function import validate  # Function to validate the model
from lib.core.loss import CrossEntropy2D  # Custom loss function for segmentation tasks
from lib.utils.utils import save_checkpoint  # Utility to save model checkpoints
from lib.utils.utils import create_logger  # Utility to create a logger for logging events

import lib.dataset as dataset  # Dataset utilities
import lib.models as models  # Model utilities

# Setting a random seed for reproducibility across runs
seed = 37
random.seed(seed)  # Set Python random seed
torch.manual_seed(seed)  # Set PyTorch seed for CPU
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # Set PyTorch seed for all available GPUs

np.random.seed(seed)  # Set NumPy random seed
os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python's hash seed for reproducibility

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train image segmentation network')

    # Directory to save outputs (default is 'out')
    parser.add_argument('--out_dir',
                        help='directory to save outputs',
                        default='out',
                        type=str)

    # How frequently to log progress (default is every 10 steps)
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=10,
                        type=int)

    # Interval for running evaluation (default is every 1 epoch)
    parser.add_argument('--eval_interval',
                        help='evaluation interval',
                        default=1,
                        type=int)

    # Whether to use GPU or not
    parser.add_argument('--gpu',
                        action='store_true',
                        help='whether to use GPU or not')

    # Number of workers for the DataLoader (default is 4)
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=4,
                        type=int)

    # Parse and return the arguments
    args = parser.parse_args()
    return args

# Main function to run the training and validation process
def main():
    args = parse_args()  # Parse the command-line arguments

    # Create a logger (both text logger and TensorBoard logger)
    logger, tb_logger = create_logger(
        args.out_dir, phase='train', create_tf_logs=True)  # TensorBoard logging

    # Log the parsed arguments
    logger.info(pprint.pformat(args))

    # Initialize the segmentation model (lite version of a segmentation network)
    model = models.seg_net_lite.get_seg_net()

    # TensorBoard writer dictionary for logging purposes
    if tb_logger:
        writer_dict = {
            'logger': tb_logger,
            'train_global_steps': 0,  # Global training steps for logging
            'valid_global_steps': 0,  # Global validation steps for logging
            'vis_global_steps': 0,  # Global visualization steps for logging
        }
    else:
        writer_dict = None

    # Define the loss function (cross-entropy) and optimizer (Adam)
    if args.gpu:
        model = model.cuda()  # Move the model to GPU if available
        criterion = CrossEntropy2D(ignore_index=255).cuda()  # Loss function, ignoring certain labels (e.g., background)
    else:
        criterion = CrossEntropy2D(ignore_index=255)  # Loss function for CPU

    optimizer = optim.Adam(model.parameters())  # Adam optimizer for model parameters

    # Load the training and validation datasets (using MNIST as an example)
    train_dataset = dataset.mnist(is_train=True)  # Load the training dataset
    val_dataset = dataset.mnist(is_train=False)  # Load the validation dataset

    # Create DataLoader for the training dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # Set batch size to 1 for simplicity
        shuffle=True,  # Shuffle the dataset for training
        num_workers=args.num_workers,  # Number of workers for data loading
        pin_memory=True  # Use pinned memory for faster data transfer to GPU
    )

    # Create DataLoader for the validation dataset
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # Set batch size to 1 for validation
        shuffle=False,  # No need to shuffle the validation data
        num_workers=1,  # Use a single worker for validation
        pin_memory=True
    )

    best_perf = 0.0  # Track the best performance (mean IoU - Intersection over Union)
    best_model = False  # Flag to track whether the current model is the best

    train_epochs = 20  # Train for 20 epochs (this can be adjusted based on need)
    for epoch in range(train_epochs):
        # Train the model for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              args.out_dir, writer_dict, args)

        # Evaluate the model at specified intervals
        if (epoch + 1) % args.eval_interval == 0:
            # Validate the model on the validation dataset
            perf_indicator = validate(val_loader, val_dataset, model,
                                      criterion, args.out_dir, writer_dict, args)

            # Update the best performance if the current performance is better
            if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False
        else:
            perf_indicator = -1  # No evaluation done this epoch
            best_model = False

        # Save the model checkpoint after every epoch
        logger.info('=> saving checkpoint to {}'.format(args.out_dir))
        save_checkpoint({
            'epoch': epoch + 1,  # Current epoch number
            'state_dict': model.state_dict(),  # Model state dictionary (weights)
            'perf': perf_indicator,  # Current performance indicator
            'last_epoch': epoch,  # Last epoch number
            'optimizer': optimizer.state_dict(),  # Optimizer state dictionary
        }, best_model, args.out_dir)

    # Save the final model state after all epochs
    final_model_state_file = os.path.join(args.out_dir, 'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)  # Save the final model state

    # Close the TensorBoard logger
    writer_dict['logger'].close()


# If this script is run directly, execute the main function
if __name__ == '__main__':
    main()
