import imageio
import os
import numpy as np
import sys
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

sys.path.append('.')  # noqa: E402
from src.recurrent_attention_network_paper.model import RACNN
from src.recurrent_attention_network_paper.CUB_loader import CUB200_loader
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def log(msg):
    open('build/core.log', 'a').write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]\t'+msg+'\n'), print(msg)


def eval(net, dataloader):
    log(' :: Testing on test set ...')
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        with torch.no_grad():
            logits = net(inputs)
            correct_top1 += torch.eq(logits.topk(max((1, 1)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()
            correct_top3 += torch.eq(logits.topk(max((1, 3)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()
            correct_top5 += torch.eq(logits.topk(max((1, 5)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()

        if step > 200:
            log(f'\tAccuracy@top1 ({step}/{len(dataloader)}) = {correct_top1/((step+1)*int(inputs.shape[0])):.5%}')
            log(f'\tAccuracy@top3 ({step}/{len(dataloader)}) = {correct_top3/((step+1)*int(inputs.shape[0])):.5%}')
            log(f'\tAccuracy@top5 ({step}/{len(dataloader)}) = {correct_top5/((step+1)*int(inputs.shape[0])):.5%}')
            return


def run():
    state_dict = torchvision.models.mobilenet.mobilenet_v2(pretrained=True).state_dict()
    state_dict.pop('classifier.1.weight')
    state_dict.pop('classifier.1.bias')
    net = torchvision.models.mobilenet.mobilenet_v2(num_classes=200).cuda()
    state_dict['classifier.1.weight'] = net.state_dict()['classifier.1.weight']
    state_dict['classifier.1.bias'] = net.state_dict()['classifier.1.bias']
    net.load_state_dict(state_dict)
    cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainset = CUB200_loader('external/CUB_200_2011', split='train')
    testset = CUB200_loader('external/CUB_200_2011', split='test')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=trainset.CUB_collate, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, collate_fn=testset.CUB_collate, num_workers=4)

    classes = [line.split(' ')[1] for line in open('external/CUB_200_2011/classes.txt', 'r').readlines()]
    log(' :: Start training ...')

    for epoch in range(100):  # loop over the dataset multiple times
        losses = 0
        for step, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses += loss
            if step % 20 == 0 and step != 0:
                avg_loss = losses/20
                log(f':: loss @step({step:2d}/{len(trainloader)})-epoch{epoch}: {loss:.10f}\tavg_loss_20: {avg_loss:.10f}')
                losses = 0
        eval(net, testloader)
        if epoch % 20 == 0 and epoch != 0:
            stamp = f'e{epoch}{int(time.time())}'
            torch.save(net, f'build/mobilenet_v2_cub200-{stamp}.pt')
            torch.save(optimizer.state_dict, f'build/optimizer-{stamp}.pt')


if __name__ == "__main__":
    path = 'build'
    if not os.path.exists(path):
        os.makedirs(path)
    run()
