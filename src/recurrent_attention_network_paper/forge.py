import sys
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('.')  # noqa: E402
from src.recurrent_attention_network_paper.model import RACNN
from src.recurrent_attention_network_paper.CUB_loader import CUB200_loader
from src.recurrent_attention_network_paper.pretrain_apn import random_sample
from torch.autograd import Variable


def avg(x): return sum(x)/len(x)


def log(msg): open('build/core.log', 'a').write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]\t'+msg+'\n'), print(msg)


def train(net, dataloader, optimizer, epoch, _type, sample):
    assert _type in ['apn', 'backbone']
    losses = []
    net.mode(_type), log(f' :: Switch to {_type}')
    for step, (inputs, targets) in enumerate(dataloader, 0):
        loss = net.echo(inputs, targets, optimizer)
        losses.append(loss)
        avg_loss = avg(losses[-5 if len(losses) > 5 else -len(losses):])
        log(f":: loss @epoch{epoch:2d}/step{step:2d} ({_type}): {loss:.12f}\tavg_loss_5: {avg_loss:.12f}")

        if step % 5 == 0 and step != 0:  # check point
            _, _, attens, resized = net(sample.unsqueeze(0))
            x1, x2 = resized[0].data, resized[1].data
            plt.imshow(CUB200_loader.tensor_to_img(x1[0]))
            plt.show()
            plt.imshow(CUB200_loader.tensor_to_img(x2[0]))
            plt.show()

        if step > 60:
            break


def test(net, dataloader, stamp):
    log(' :: Testing on test set ...')
    with torch.no_grad():
        net.eval()
        corrects1 = 0
        corrects2 = 0
        corrects3 = 0
        cnt = 0
        test_cls_losses = []
        test_apn_losses = []
        for idx, (test_images, test_labels) in enumerate(dataloader, 0):
            if idx % 20 == 0:
                log(f' :: Inferencing test sample ({idx}/{len(dataloader)})')

            test_images = test_images.cuda()
            test_labels = test_labels.cuda()
            cnt += test_labels.size(0)
            preds, _, _, _ = net(test_images)
            test_cls_loss = net.multitask_loss(preds, test_labels)
            test_apn_loss = net.rank_loss(preds, test_labels)
            test_cls_losses.append(test_cls_loss)
            test_apn_losses.append(test_apn_loss)
            _, predicted1 = torch.max(preds[0], 1)
            correct1 = (predicted1 == test_labels).sum()
            corrects1 += correct1
            _, predicted2 = torch.max(preds[1], 1)
            correct2 = (predicted2 == test_labels).sum()
            corrects2 += correct2
            _, predicted3 = torch.max(preds[2], 1)
            correct3 = (predicted3 == test_labels).sum()
            corrects3 += correct3

        test_cls_losses = torch.stack(test_cls_losses).mean()
        test_apn_losses = torch.stack(test_apn_losses).mean()
        accuracy1 = corrects1.float() / cnt
        accuracy2 = corrects2.float() / cnt
        accuracy3 = corrects3.float() / cnt
        log(f'test_cls_loss:\t{test_cls_losses.item()} ({stamp})')
        log(f'test_rank_loss:\t{test_apn_losses.item()} ({stamp})')
        log(f'test_acc1:\t{accuracy1.item()} ({stamp})')
        log(f'test_acc2:\t{accuracy2.item()} ({stamp})')
        log(f'test_acc3:\t{accuracy3.item()} ({stamp})')


def run():
    net = RACNN(num_classes=200).cuda()
    cudnn.benchmark = True

    head_params = list(net.classifier1.parameters()) + list(net.classifier2.parameters()) + list(net.classifier3.parameters())
    cls_params = list(net.b1.parameters()) + list(net.b2.parameters()) + list(net.b3.parameters()) + \
        list(net.classifier1.parameters()) + list(net.classifier2.parameters()) + list(net.classifier3.parameters())
    apn_params = list(net.apn1.parameters()) + list(net.apn2.parameters())

    head_opt = optim.SGD(head_params, lr=0.001, momentum=0.9)
    cls_opt = optim.SGD(cls_params, lr=0.001, momentum=0.9)
    apn_opt = optim.SGD(apn_params, lr=0.001, momentum=0.9)

    trainset = CUB200_loader('external/CUB_200_2011', split='train')
    testset = CUB200_loader('external/CUB_200_2011', split='test')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=trainset.CUB_collate, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, collate_fn=testset.CUB_collate, num_workers=4)
    sample = random_sample(testloader)

    net.load_state_dict(torch.load('build/racnn_pretrained.pt'))

    epoch = 0
    # train(net, trainloader, head_opt, epoch, 'backbone', sample)
    # train(net, trainloader, cls_opt, epoch, 'backbone', sample)
    train(net, trainloader, apn_opt, epoch, 'apn', sample)
    # test(net, testloader, 'head fine-tune')



if __name__ == "__main__":
    run()
