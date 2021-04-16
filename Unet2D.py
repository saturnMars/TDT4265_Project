import torch
from torch import nn

class Unet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        num_channels = 32
        
        self.conv1 = self.contract_block(in_channels, num_channels, 3, 1)
        self.conv2 = self.contract_block(num_channels, num_channels * 2, 3, 1)
        self.conv3 = self.contract_block(num_channels * 2, num_channels * 4, 3, 1)
        self.conv4 = self.contract_block(num_channels * 4, num_channels * 8, 3, 1)
        self.conv5 = self.contract_block(num_channels * 8, num_channels * 16, 3, 1)
        
        self.upconv5 = self.expand_block(num_channels * 16, num_channels * 8, 3, 1)
        self.upconv4 = self.expand_block(num_channels * 16, num_channels * 4, 3, 1)
        self.upconv3 = self.expand_block(num_channels * 8, num_channels * 2, 3, 1)
        self.upconv2 = self.expand_block(num_channels * 4, num_channels, 3, 1)
    
        self.upconv1 = self.last_block(num_channels * 2, out_channels, 3, 1)
        

    def __call__(self, x):
        # Downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Upsample
        upconv5 = self.upconv5(conv5)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], dim=1))
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], dim=1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], dim=1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], dim=1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        layers = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout2d(p=0.2),
            
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout2d(p=0.1),
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ]
            
        contract = nn.Sequential(*layers)
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        layers = [
            torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout2d(p=0.2),
            
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout2d(p=0.1),
        ]
    
        expand = nn.Sequential(*layers)
        return expand
    
    def last_block(self, in_channels, out_channels, kernel_size, padding):
        middle_channels = in_channels//2
        
        layers = [
            torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            
            torch.nn.Conv2d(in_channels, middle_channels , kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(middle_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout2d(p=0.2),
            
            torch.nn.Conv2d(middle_channels, middle_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(middle_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout2d(p=0.2),
            
            torch.nn.Conv2d(middle_channels, middle_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(middle_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout2d(p=0.1),
            
            torch.nn.Conv2d(middle_channels, out_channels, 1),
            torch.nn.Softmax(dim=1)
        ]
        
        last = nn.Sequential(*layers)
        return last