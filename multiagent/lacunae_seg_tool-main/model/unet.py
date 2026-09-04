"""
Utility functions and UNet CNN definition. 
The model provided allows for 63.92 (4.69) [%] DSC performance.
"""

import torch
import torch.nn as nn

# Utility Functions

def conv_bn_leru(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """
    Block of 2 Conv2 + BatchNorm2d + ReLU
    """
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
    )

def down_pooling():
    """
    Max pooling operation for encoder downscaling
    """
    return nn.MaxPool2d(2)

def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    """
    UNet block for decoder upsampling 
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# UNet neural network architecture
class UNet(nn.Module):
    """
    Standard UNet neural network architecture as described in: 
    Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." 
    Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, 
    October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015.
    """
    def __init__(self, input_channels, nclasses):
        super().__init__()
        
        # go down
        self.conv1 = conv_bn_leru(input_channels,16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.down_pooling = nn.MaxPool2d(2)

        # go up
        self.up_pool6 = up_pooling(256, 128)
        self.conv6 = conv_bn_leru(256, 128)
        
        self.up_pool7 = up_pooling(128, 64)
        self.conv7 = conv_bn_leru(128, 64)
        
        self.up_pool8 = up_pooling(64, 32)
        self.conv8 = conv_bn_leru(64, 32)
        
        self.up_pool9 = up_pooling(32, 16)
        self.conv9 = conv_bn_leru(32, 16)

        self.conv10 = nn.Conv2d(16, nclasses, 1)

    def forward(self, x):
        # go down
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        
        x5 = self.conv5(p4)

        # go up
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = torch.sigmoid(output)

        return output

