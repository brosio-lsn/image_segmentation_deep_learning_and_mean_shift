from __future__ import absolute_import, division, print_function
# These imports ensure compatibility between Python 2 and Python 3 for important language features.

import numpy as np  # For numerical operations (though not used here)

import torch.nn as nn  # PyTorch's neural network module

# Helper function to ensure the target tensor does not require gradients
def _assert_no_grad(tensor):
    """
    Ensure that the target tensor does not require gradients.
    PyTorch loss functions do not compute gradients with respect to the target.
    Args:
        tensor (torch.Tensor): Input tensor (usually ground truth labels).
    Raises:
        AssertionError: If the tensor requires gradients.
    """
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

# Custom loss function: CrossEntropy2D
class CrossEntropy2D(nn.Module):
    def __init__(self, ignore_index, reduction='mean', weight=None):
        """
        Initialize the CrossEntropy2D loss function, which extends PyTorch's cross-entropy
        to handle 2D outputs (such as in semantic segmentation tasks).
        
        Args:
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            reduction (str): Specifies the reduction to apply to the output ('none', 'mean', 'sum'). Defaults to 'mean'.
            weight (torch.Tensor, optional): A manual rescaling weight given to each class. Defaults to None.
        """
        super(CrossEntropy2D, self).__init__()
        self.weight = weight  # Class weights, if provided
        self.ignore_index = ignore_index  # Index to ignore in the loss calculation (e.g., background class)
        self.reduction = reduction  # Reduction method ('none', 'mean', 'sum')

    def forward(self, output, target, resize_scores=True):
        """
        Forward pass of the loss function. Computes the cross-entropy loss between the model's predictions
        and the ground truth target, handling cases where the input and target dimensions differ.
        
        Args:
            output (torch.Tensor): The model's raw output logits (before applying softmax).
            target (torch.Tensor): Ground truth labels (integer labels for each pixel).
            resize_scores (bool): Whether to resize the output logits to match the target size (default is True).
        
        Returns:
            loss (torch.Tensor): The computed cross-entropy loss.
        """
        _assert_no_grad(target)  # Ensure the target tensor does not require gradients

        # Get the dimensions of the output and target tensors
        b, c, h, w = output.size()  # Output: [batch_size, num_classes, height, width]
        tb, th, tw = target.size()  # Target: [batch_size, height, width]
        
        assert(b == tb)  # Ensure the batch size is the same

        # Handle cases where the output and target sizes don't match
        if resize_scores:
            if h != th or w != tw:  # If height/width mismatch, upsample the output to match target size
                output = nn.functional.interpolate(output, size=(th, tw), mode="bilinear", align_corners=False)
        else:
            if h != th or w != tw:  # If resizing is not allowed, downsample the target to match output size
                target = nn.functional.interpolate(
                    target.view(b, 1, th, tw).float(), size=(h, w), mode="nearest"
                ).view(b, h, w).long()

        # Compute the cross-entropy loss between the output and target
        loss = nn.functional.cross_entropy(
            output, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )

        return loss  # Return the computed loss
