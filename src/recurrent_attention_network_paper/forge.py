import imageio
import os
import shutil
import sys
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

sys.path.append('.')  # noqa: E402
from src.recurrent_attention_network_paper.model import RACNN
from src.recurrent_attention_network_paper.CUB_loader import CUB200_loader
from src.recurrent_attention_network_paper.pretrain_apn import random_sample, save_img, clean, log, build_gif



def avg(x): return sum(x)/len(x)


def train(net, dataloader, optimizer, epoch, _type):
    assert _type in ['apn', 'backbone']
    losses = 0
    net.mode(_type), log(f' :: Switch to {_type}')  # switch loss type
    for step, (inputs, targets) in enumerate(dataloader, 0):
        loss = net.echo(inputs, targets, optimizer)
        losses += loss

        if step % 20 == 0 and step != 0:
            avg_loss = losses/20
            log(f':: loss @step({step:2d}/{len(dataloader)})-epoch{epoch}: {loss:.10f}\tavg_loss_20: {avg_loss:.10f}')
            losses = 0

    return avg_loss


def test(net, dataloader):
    log(' :: Testing on test set ...')
    correct_summary = {'clsf-0': {'top-1': 0, 'top-5': 0}, 'clsf-1': {'top-1': 0, 'top-5': 0}, 'clsf-2': {'top-1': 0, 'top-5': 0}}
    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        with torch.no_grad():
            outputs, _, _, _ = net(inputs)
            for idx, logits in enumerate(outputs):
                correct_summary[f'clsf-{idx}']['top-1'] += torch.eq(logits.topk(max((1, 1)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()  # top-1
                correct_summary[f'clsf-{idx}']['top-5'] += torch.eq(logits.topk(max((1, 5)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()  # top-5

            if step > 200:
                for clsf in correct_summary.keys():
                    _summary = correct_summary[clsf]
                    for topk in _summary.keys():
                        log(f'\tAccuracy {clsf}@{topk} ({step}/{len(dataloader)}) = {_summary[topk]/((step+1)*int(inputs.shape[0])):.5%}')
                return


def run(pretrained_model):
    log(f' :: Start training with {pretrained_model}')
    net = RACNN(num_classes=200).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    cudnn.benchmark = True

    cls_params = list(net.b1.parameters()) + list(net.b2.parameters()) + list(net.b3.parameters()) + \
        list(net.classifier1.parameters()) + list(net.classifier2.parameters()) + list(net.classifier3.parameters())
    apn_params = list(net.apn1.parameters()) + list(net.apn2.parameters())

    cls_opt = optim.SGD(cls_params, lr=0.001, momentum=0.9)
    apn_opt = optim.SGD(apn_params, lr=0.001, momentum=0.9)

    trainset = CUB200_loader('external/CUB_200_2011', split='train')
    testset = CUB200_loader('external/CUB_200_2011', split='test')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=trainset.CUB_collate, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, collate_fn=testset.CUB_collate, num_workers=4)
    sample = random_sample(testloader)

    for epoch in range(50):
        cls_loss = train(net, trainloader, cls_opt, epoch, 'backbone')
        rank_loss = train(net, trainloader, apn_opt, epoch, 'apn')
        test(net, testloader)

        # visualize cropped inputs
        _, _, _, resized = net(sample.unsqueeze(0))
        x1, x2 = resized[0].data, resized[1].data
        save_img(x1, path=f'build/.cache/epoch_{epoch}@2x.jpg', annotation=f'cls_loss = {cls_loss:.7f}, rank_loss = {rank_loss:.7f}')
        save_img(x2, path=f'build/.cache/epoch_{epoch}@4x.jpg', annotation=f'cls_loss = {cls_loss:.7f}, rank_loss = {rank_loss:.7f}')

        # save model per 10 epoches
        if epoch % 10 == 0 and epoch != 0:
            stamp = f'e{epoch}{int(time.time())}'
            torch.save(net.state_dict, f'build/racnn_mobilenetv2_cub200-e{epoch}s{stamp}.pt')
            log(f' :: Saved model dict as:\tbuild/racnn_mobilenetv2_cub200-e{epoch}s{stamp}.pt')
            torch.save(cls_opt.state_dict(), f'build/cls_optimizer-s{stamp}.pt')
            torch.save(apn_opt.state_dict(), f'build/apn_optimizer-s{stamp}.pt')


if __name__ == "__main__":
    clean()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(pretrained_model='build/racnn_pretrained-1577771401.pt')
    build_gif(pattern='@2x', gif_name='racnn_cub200')
    build_gif(pattern='@4x', gif_name='racnn_cub200')
