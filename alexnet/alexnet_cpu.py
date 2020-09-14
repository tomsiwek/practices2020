import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import time


def to_categorical(nums, ncat):
    out_tensor = torch.zeros(len(nums), ncat)
    for i, num in enumerate(nums):
        out_tensor[i, num] = 1
    return out_tensor

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Downloading training data
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle=True, num_workers=2)

#Downloading test data
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

testloader = torch.utils.data.DataLoader(test_data, batch_size=40, shuffle=False, num_workers=2)

#Class labels
classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

device = torch.device("hip:0")

#sol.device.set(sol.device.ve, 0)

#class TrainingModel(torch.nn.Module):
#    def __init__(self, model):
#        super().__init__()
#        self.m_model = model
#        self.m_model.classifier[4] = nn.Linear(4096, 1024)
#        self.m_model.classifier[6] = nn.Linear(1024, 10)
#        self.m_loss  = nn.L1Loss()

#   def forward(self, x, target):
#        output = self.m_model(x)
#        loss   = self.m_loss(output, target)
#        return (output, loss)
#
py_model = models.__dict__["alexnet"]()
py_model.classifier[4] = nn.Linear(4096, 1024)
py_model.classifier[6] = nn.Linear(1024, 10)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(py_model.parameters(), lr=.01, momentum=.9)

start_time = time.time()

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data #data[0].to(device), data[1].to(device)
        
        #labels = to_categorical(labels, 10)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = py_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            #print('[%d, %5d] loss: %.3f' %
            #    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

elapsed_time = time.time() - start_time

#print('Finished Training')
print("Elapsed time[s]: %.2f"%elapsed_time)
