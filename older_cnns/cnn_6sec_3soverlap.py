import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchsummary import summary
import time

from IPython.display import HTML, display

start_time = time.time()
train_times = []
val_times = []


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
    transforms.ToTensor()
    #transforms.CenterCrop([369, 375])
    # ,
    # transforms.Resize([256,256])
    # transforms.RandomCrop([288,50])
    # transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
    # (0.24703233, 0.24348505, 0.26158768))
])

haveCuda = torch.cuda.is_available()

trainSet = torchvision.datasets.ImageFolder(root="C:/Users/beni1/Desktop/Önlab/spectograms6s_overlap3s/train/", transform=transform)
testSet = torchvision.datasets.ImageFolder(root="C:/Users/beni1/Desktop/Önlab/spectograms6s_overlap3s/test/", transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=128, shuffle=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=128, shuffle=False)

net = NETWORK2()
print(net)
summary(net.cuda(),(3,275,275))
best_train_acc = 0
best_val_acc = 0

print("Initialization took %.2f seconds." % (time.time()-start_time))


def train(epoch):

    # variables for loss
    running_loss = 0.0
    correct = 0.0
    total = 0
    train_start_time = time.time()

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
    train_time = time.time()-train_start_time
    train_times.append(train_time)
    global best_train_acc
    if tr_corr > best_train_acc:
        best_train_acc = tr_corr
    print("Train epoch %d loss: %.3f correct: %.2f duration: %.2f s" % (epoch + 1, tr_loss, tr_corr,train_time))

    return tr_loss,tr_corr


def val(epoch):

    # variables for loss
    running_loss = 0.0
    correct = 0.0
    total = 0
    val_start_time = time.time()

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
    val_time = time.time()-val_start_time
    val_times.append(val_time)
    global best_val_acc
    if val_corr > best_val_acc:
        best_val_acc = val_corr
    print("Test  epoch %d loss: %.3f correct: %.2f duration: %.2f s" % (epoch + 1, val_loss, val_corr,val_time))
    print("-------------------------------------------------------")
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
numEpoch = 35

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
        #torch.save(net, "/bestModel_vXX.pth")
    # Step with the scheduler
    scheduler.step()

# Finished
print('Finished Training')
print("Best training accuracy: %.2f" % (best_train_acc))
print("Best validation accuracy: %.2f" % (best_val_acc))
print("Avg train time: %.2f" % (sum(train_times)/len(train_times)))
print("Avg val time  : %.2f" % (sum(val_times)/len(val_times)))
print("Total duration: %.2f" % (time.time()-start_time))
plt.plot(trLosses)
plt.plot(valLosses)
plt.show()
plt.plot(trAccs)
plt.plot(valAccs)
plt.show()

net.eval()

conf = torch.zeros(9,9)

# Create progress bar

# Epoch loop
for i, data in enumerate(testLoader, 0):
    # get the inputs
    inputs, labels = data

    # Convert to cuda conditionally
    if haveCuda:
        inputs, labels = inputs.cuda(), labels.cuda()

    # forward
    outputs = net(inputs)

    # compute statistics
    _, predicted = torch.max(outputs, 1)
    for label,pred in zip (labels,predicted):
        conf[label,pred] += 1

    # Update progress bar
print(conf)

