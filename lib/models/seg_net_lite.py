from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        layers_conv_down = []
        layers_bn_down = []
        layers_pooling = []
        for i in range(self.num_down_layers):
            layers_conv_down.append(nn.Conv2d(input_size, down_filter_sizes[i], kernel_sizes[i], padding=conv_paddings[i]))
            layers_bn_down.append(nn.BatchNorm2d(down_filter_sizes[i]))
            layers_pooling.append(nn.MaxPool2d(pooling_kernel_sizes[i], stride=pooling_strides[i], return_indices=True))
            input_size = down_filter_sizes[i]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = []
        layers_bn_up = []
        layers_unpooling = []
        for i in range(self.num_up_layers):
            layers_unpooling.append(nn.MaxUnpool2d(pooling_kernel_sizes[i], stride=pooling_strides[i]))
            layers_conv_up.append(nn.Conv2d(input_size, up_filter_sizes[i], kernel_sizes[i], padding=conv_paddings[i]))
            layers_bn_up.append(nn.BatchNorm2d(up_filter_sizes[i]))
            input_size = up_filter_sizes[i]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)
        

        self.relu = nn.ReLU(True)

        self.conv_final = nn.Conv2d(input_size, 11, 1, padding=0)

    def forward(self, x):
        #Forward pass of the SegNet module
        # x: input image
        # return: logits of segmentation output
        indices_list = []
        size_list = []

        # Downsampling
        for conv, bn, pool in zip(self.layers_conv_down, self.layers_bn_down, self.layers_pooling):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            size_list.append(x.size())
            x, indices = pool(x)
            indices_list.append(indices)

        # Upsampling
        for unpool, conv, bn in zip(self.layers_unpooling, self.layers_conv_up, self.layers_bn_up):
            indices = indices_list.pop()
            size = size_list.pop()
            x = unpool(x, indices, output_size=size)
            x = conv(x)
            x = bn(x)
            x = self.relu(x)

        # Final convolution
        x = self.conv_final(x)
        return x

        


def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
