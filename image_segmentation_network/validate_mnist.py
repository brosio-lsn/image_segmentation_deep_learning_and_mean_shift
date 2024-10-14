from __future__ import absolute_import, division, print_function
# These imports ensure compatibility between Python 2 and Python 3 for important language features.

import argparse  # For parsing command-line arguments
import os  # For interacting with the file system
import pprint  # Pretty-print for better formatted logging

import torch  # Main PyTorch library
import torch.optim as optim  # PyTorch optimizers
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

    # Whether to use GPU or not
    parser.add_argument('--gpu',
                        action='store_true',
                        help='whether to use GPU or not')

    # Number of workers for the DataLoader (default is 1)
    parser.add_argument('--num_workers',
                        help='num of dataloader workers',
                        default=1,
                        type=int)

    # Parse and return the arguments
    args = parser.parse_args()
    return args

# Main function to run the validation process
def main():
    args = parse_args()  # Parse the command-line arguments

    # Create a logger (both text logger and TensorBoard logger)
    logger, tb_logger = create_logger(
        args.out_dir, phase='valid', create_tf_logs=True)  # TensorBoard logging

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

    # Define the loss function (cross-entropy) and move the model to the appropriate device
    if args.gpu:
        model = model.cuda()  # Move the model to GPU if available
        criterion = CrossEntropy2D(ignore_index=255).cuda()  # Loss function, ignoring certain labels (e.g., background)
    else:
        criterion = CrossEntropy2D(ignore_index=255)  # Loss function for CPU

    # Load the best model from the specified output directory
    model_state_file = os.path.join(args.out_dir, 'model_best.pth.tar')  # Path to the best model file
    logger.info('=> loading model from {}'.format(model_state_file))  # Log the loading process
    state_dict = torch.load(model_state_file, map_location=torch.device('cpu'))  # Load model to CPU (or GPU if applicable)
    model.load_state_dict(state_dict)  # Load the saved state dictionary into the model

    # Load the validation dataset (MNIST in this case)
    val_dataset = dataset.mnist(is_train=False)  # Load the validation dataset (test split)

    # Create a DataLoader for the validation dataset
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # Set batch size to 1 for simplicity
        shuffle=False,  # No need to shuffle validation data
        num_workers=args.num_workers,  # Number of workers for data loading
        pin_memory=True  # Use pinned memory for faster data transfer to GPU
    )

    # Evaluate the model on the validation set
    perf_indicator = validate(val_loader, val_dataset, model,
                              criterion, args.out_dir, writer_dict, args)

    # Close the TensorBoard logger
    writer_dict['logger'].close()


# If this script is run directly, execute the main function
if __name__ == '__main__':
    main()
