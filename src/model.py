import torch
import torch.nn as nn
import torchvision.models as models

class PCOSResNet18(nn.Module):
    def __init__(self):
        super(PCOSResNet18, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=original_conv.bias
        )
        
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        for param in self.resnet.conv1.parameters():
            param.requires_grad = True

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 2) 
        )

    def forward(self, x):
        return self.resnet(x)