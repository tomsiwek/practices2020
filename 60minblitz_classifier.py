import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sol.pytorch as sol
import time

transform   = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
trainset    = torchvision.datasets.CIFAR10(root="./data", train=True,
                                           download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset     = torchvision.datasets.CIFAR10(root="./data", train=False,
                                           download=False, transform=transform)
testloader  = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# sol.device.set(sol.device.ve, 0) # not needed in native offloading

device = torch.device("hip:0")

print("Train data shape:", trainset.data.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=.001, momentum=.9)


# input_dummy = torch.rand(50000, 3, 32, 32) # no need to explicitly allocate data for this
opt=sol.optimize(net, sol.input([0, 3, 32, 32], batch_size=32)) # 0 == wildcard if batchsize is unknown when sol.optimize gets called, or if it varies during training
opt.load_state_dict(net.state_dict(), strict=False)
opt.to(device) # copy to VE

start_time = time.time()

opt.train()
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data #data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
		inputs  = inputs.to(device) # copy to VE
        outputs = net(inputs)
		outputs = outputs.cpu() # copy back to CPU
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

elapsed_time = time.time() - start_time

print('Finished Training')
print("Elapsed time[s]: %.2f"%elapsed_time)

correct = 0
total   = 0
opt.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
		images = images.to(device)
		labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1) # is the outputs.data correct?
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))
