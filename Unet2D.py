import torch
from torch import nn

class Unet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        num_channels = 32
        self.conv1 = self.contract_block(in_channels, num_channels, 7, 3)
        self.conv2 = self.contract_block(num_channels, num_channels * 2, 3, 1)
        self.conv3 = self.contract_block(num_channels * 2, num_channels * 4, 3, 1)
        self.conv4 = self.contract_block(num_channels * 4, num_channels * 8, 3, 1)
        self.conv5 = self.contract_block(num_channels * 8, num_channels * 16, 3, 1)
        
        # ORIGINAL MODEL: 
        # L1: 128 - 64
        # L2: 64*2 - 32
        # L3: 32*2 - output_channels
        
        # NEW MODEL: 
        # L1: 512 - 256
        # L2: 512 - 128
        # L3: 256 - 64
        # L4: 128 - 32
        # L5: 64 - output_channels
        
        self.upconv5 = self.expand_block(num_channels * 16, num_channels * 8, 3, 1)
        self.upconv4 = self.expand_block(num_channels * 16, num_channels * 4, 3, 1
        self.upconv3 = self.expand_block(num_channels * 8, num_channels * 2, 3, 1)
        self.upconv2 = self.expand_block(num_channels * 4, num_channels, 3, 1)
        self.upconv1 = self.expand_block(num_channels * 2, out_channels, 3, 1)

    def __call__(self, x):
        # Downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv3(conv3)
        conv5 = self.conv3(conv4)

        # Upsample
        upconv5 = self.upconv5(conv5)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))
        upconv3 = self.upconv3(torch.cat([upconv4, conv3, 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2, 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            torch.nn.Dropout(p=0.1)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),

                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),

                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                            torch.nn.Dropout(p=0.1)
                              )
        return expand
