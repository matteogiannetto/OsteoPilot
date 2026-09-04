import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, dropout_rate=0):
        super(U_Net, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.Conv1 = conv_block(in_ch, filters[0], dropout_rate)
        self.Conv2 = conv_block(filters[0], filters[1], dropout_rate)
        self.Conv3 = conv_block(filters[1], filters[2], dropout_rate)
        self.Conv4 = conv_block(filters[2], filters[3], dropout_rate)
        self.Conv5 = conv_block(filters[3], filters[4], dropout_rate)
        
        # Decoder
        self.Up5 = up_conv(filters[4], filters[3], dropout_rate)
        self.Up_conv5 = conv_block(filters[4], filters[3], dropout_rate)
        
        self.Up4 = up_conv(filters[3], filters[2], dropout_rate)
        self.Up_conv4 = conv_block(filters[3], filters[2], dropout_rate)
        
        self.Up3 = up_conv(filters[2], filters[1], dropout_rate)
        self.Up_conv3 = conv_block(filters[2], filters[1], dropout_rate)
        
        self.Up2 = up_conv(filters[1], filters[0], dropout_rate)
        self.Up_conv2 = conv_block(filters[1], filters[0], dropout_rate)
        
        self.Conv_1x1 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        e1 = self.Conv1(x)
        
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)
        
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        output = self.Conv_1x1(d2)
        
        return torch.sigmoid(output)  # Keep for binary segmentation
    
def init_weights(model, init_type='xavier_uniform'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            else:
                raise ValueError("Unsupported initialization type: choose 'xavier_uniform' or 'xavier_normal'")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # For batch norm layers, it's common to initialize the weights to 1 and biases to 0
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        