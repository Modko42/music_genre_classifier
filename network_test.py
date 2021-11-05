import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np

class NETWORK2(nn.Module):
    def __init__(self):
        super(NETWORK2, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
        )
        #15488


        self.flatten = nn.Flatten()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear_layers = nn.Linear(128, 9)
        #1936
        #self.classifier = nn.Conv2d(15488, 9, kernel_size=1)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        #x = x.view(x.size(0), -1)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        #x = self.classifier(x)
        return x





net = torch.load("bestModel_v2.pth")

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop([369, 375])
    # ,
    # transforms.Resize([256,256])
    # transforms.RandomCrop([288,50])
    # transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
    # (0.24703233, 0.24348505, 0.26158768))
])

haveCuda = torch.cuda.is_available()

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop([369, 375])
])


net.eval()
image_loaded = image_loader(data_transforms,"country77_noalpha.png")

image_loaded = image_loaded.cuda()

output = net(image_loaded).cpu().detach().numpy()
norm = np.linalg.norm(output)
output = output / norm

print(output)