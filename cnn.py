import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchsummary import summary

from IPython.display import HTML, display

def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

print(torch.cuda.is_available())

class Conv(nn.Module):
  def __init__(self, inplanes, planes, size=3, stride=1,nLinType=1,bNorm=True):
    super(Conv, self).__init__()
    self.conv = nn.Conv2d(inplanes, planes, kernel_size=size, padding=size // 2, stride=stride)
    self.bn = nn.BatchNorm2d(planes)
    self.nLinType = nLinType
    self.bNorm = bNorm
  def forward(self, x):
    if self.nLinType == 1:
      if self.bNorm:
        return self.bn(torch.relu(self.conv(x)))
      else:
        return torch.relu(self.conv(x))
    else:
       if self.bNorm:
        return self.bn(torch.sigmoid(self.conv(x)))
       else:
        return torch.sigmoid(self.conv(x))
class Level(nn.Module):
    def __init__(self, numOfLayers, inplanes, planes, kernelSize, nLinType_, bNorm_,residual_):
      super().__init__()
      layers = []
      self.residual = residual_
      for i in range(numOfLayers):
        layers.append(Conv(inplanes,planes,kernelSize,nLinType = nLinType_,bNorm=bNorm_))

      layers.append(Conv(inplanes,planes*2,kernelSize,stride=2,nLinType = nLinType_,bNorm=bNorm_))
      self.layers = nn.Sequential(*layers)
    def forward(self, x):
        if self.residual:
             print(x.shape)
             out = self.layers[:-1].forward(x) + x
             print(out.shape)
             return self.layers[-1].forward(out)
        return self.layers.forward(x)

class NETWORK(nn.Module):
    def __init__(self,numClass,nFeat,nLevels,layersPerLevel,kernelSize,nLinType,bNorm,residual=False):
        super().__init__()
        input_channels = 3
        output_channels = nFeat
        levels = []
        for i in range(nLevels):
          levels.append(Level(layersPerLevel,input_channels,output_channels,kernelSize,nLinType,bNorm,residual))
          input_channels=input_channels*2
          output_channels=output_channels*2
        self.levels = nn.Sequential(*levels)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Conv2d(output_channels, numClass, 5, padding=2)
    def forward(self, x):
        levels_output = self.levels.forward(x)
        pooled = self.pooling(levels_output)
        return torch.squeeze(self.classifier(pooled))

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

trainSet = torchvision.datasets.ImageFolder(root="C:/content2/spectrograms3sec/train/", transform=transform)
testSet = torchvision.datasets.ImageFolder(root="C:/content2/spectrograms3sec/test/", transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=128, shuffle=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=128, shuffle=False)

#net = NETWORK(9,3,3,3,5,1,True)
net = NETWORK2()
print(net)
summary(net.cuda(),(3,369,375))

def train(epoch):

    # variables for loss
    running_loss = 0.0
    correct = 0.0
    total = 0

    # set the network to train (for batchnorm and dropout)
    net.train()

    # Create progress bar

    # Epoch loop
    for i, data in enumerate(trainLoader, 0):
        # get the inputs
        inputs, labels = data

        if haveCuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # compute statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar

    # print and plot statistics
    tr_loss = running_loss / len(trainLoader)
    tr_corr = correct / total * 100
    print("Train epoch %d loss: %.3f correct: %.2f" % (epoch + 1, tr_loss, tr_corr))

    return tr_loss,tr_corr


def val(epoch):

    # variables for loss
    running_loss = 0.0
    correct = 0.0
    total = 0

    # set the network to eval (for batchnorm and dropout)
    net.eval()

    # Create progress bar

    # Epoch loop
    for i, data in enumerate(testLoader, 0):
        # get the inputs
        inputs, labels = data

        if haveCuda:
             inputs, labels = inputs.cuda(), labels.cuda()

        # forward
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # compute statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar

    # print and plot statistics
    val_loss = running_loss / len(testLoader)
    val_corr = correct / total * 100
    print("Test epoch %d loss: %.3f correct: %.2f" % (epoch + 1, val_loss, val_corr))

    return val_loss, val_corr


import torch.optim as optim
from torch.optim import lr_scheduler

import matplotlib
import matplotlib.pyplot as plt

# Makes multiple runs comparable
torch.manual_seed(42)
if haveCuda:
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# numClass,nFeat,nLevels,layersPerLevel,kernelSize,nLinType,bNorm,residual=False):
if haveCuda:
    net = net.cuda()

# Loss, and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                      nesterov=True, weight_decay=1e-4)

# Create LR cheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 10)

# Epoch counter
numEpoch = 31

trLosses = []
trAccs = []
valLosses = []
valAccs = []
best_loss = 10

for epoch in range(numEpoch):
    # Call train and val
    tr_loss, tr_corr = train(epoch)
    val_loss, val_corr = val(epoch)

    trLosses.append(tr_loss)
    trAccs.append(tr_corr)
    valLosses.append(val_loss)
    valAccs.append(val_corr)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(net, "bestModel_v2.pth")
    # Step with the scheduler
    scheduler.step()

# Finished
print('Finished Training')
plt.plot(trLosses)
plt.plot(valLosses)
plt.show()
plt.plot(trAccs)
plt.plot(valAccs)
plt.show()

torch.save(net,"bestModel_v2.pth")
