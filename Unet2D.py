import torch
from torch import nn

class Unet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        num_channels = 16
        
        self.conv1 = self.contract_block(in_channels, num_channels, 3, 1)
        self.conv2 = self.contract_block(num_channels,um_channels * 2 n, 3, 1)
        self.conv3 = self.contract_block(num_channels * 2, num_channels * 4, 3, 1)
        self.conv4 = self.contract_block(num_channels * 4, num_channels * 8, 3, 1)
        self.conv5 = self.contract_block(num_channels * 8, num_channels * 16, 3, 1, last=True)
                
        # NEW MODEL: 
        # L1: 512 - 256
        # L2: 512 - 128
        # L3: 256 - 64
        # L4: 128 - 32
        # L5: 64 - output_channels
        
        self.upconv5 = self.expand_block(num_channels * 16, num_channels * 8, 3, 1)
        self.upconv4 = self.expand_block(num_channels * 16, num_channels * 4, 3, 1)
        self.upconv3 = self.expand_block(num_channels * 8, num_channels * 2, 3, 1)
        self.upconv2 = self.expand_block(num_channels * 4, num_channels, 3, 1)
        
        self.upconv1 = self.expand_block(num_channels * 2, out_channels, 3, 1, last=True)

    def __call__(self, x):
        # Downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Upsample
        upconv5 = self.upconv5(conv5)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding, last=False):
        layers = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        ]
        
        if not last:
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2)) # padding=1
            layers.append(torch.nn.Dropout2d(p=0.1))
            
        contract = nn.Sequential(*layers)
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding, last=False):
        layers = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        ]
        
        if not last:
            layers.append(torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            layers.append(torch.nn.Dropout2d(p=0.1))
        else:
            layers.append(torch.nn.Conv2d(out_channels, out_channels, 1, stride=1))
            #layers.append(torch.nn.Softmax(dim=1))
                    
        expand = nn.Sequential(*layers)
        return expand
