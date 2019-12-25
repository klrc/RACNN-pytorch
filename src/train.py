import torch.optim as optim
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.autograd import Variable
import time


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = torchvision.datasets.CIFAR100(root='external/CIFAR100', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='external/CIFAR100', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

model = torchvision.models.mobilenet_v2(pretrained=True).cuda()
# model = torchvision.models.resnet50(pretrained=True).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def log(msg):
    open('external/training.log', 'a').write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]\t'+msg+'\n'), print(msg)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss, _interval = 0.0, 20
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % _interval == _interval-1:    # print every _interval mini-batches
            log(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / _interval:.3f}')
            running_loss = 0.0

torch.save(model, f'external/resnet50:{time.strftime("%Y%m%d:%H:%M:%S", time.localtime())}.pt'), log('Finished Training')



