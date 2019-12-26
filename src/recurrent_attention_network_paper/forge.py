import imageio
import os
import shutil
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def avg(x): return sum(x)/len(x)


def log(msg): open('build/core.log', 'a').write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]\t'+msg+'\n'), print(msg)


def train(net, dataloader, optimizer, epoch, _type, sample=None):
    assert _type in ['apn', 'backbone']
    losses = 0
    net.mode(_type), log(f' :: Switch to {_type}')
    for step, (inputs, targets) in enumerate(dataloader, 0):
        loss = net.echo(inputs, targets, optimizer)
        losses += loss

        if step % 20 == 0 and step != 0:
            avg_loss = losses/20
            log(f':: loss @step({step:2d}/{len(dataloader)})-epoch{epoch}: {loss:.10f}\tavg_loss_20: {avg_loss:.10f}')
            losses = 0

        if step > 100:
            return 0

        if step % 3 == 0 or step < 5:
            cls_loss = loss if _type == 'backbone' else 0
            rank_loss = loss if _type == 'apn' else 0
            eval_attention_sample(net, sample, cls_loss, rank_loss, step)
    return avg_loss


def eval_attention_sample(net, sample, cls_loss, rank_loss, epoch):
    _, _, attens, resized = net(sample.unsqueeze(0))
    x1, x2 = resized[0].data, resized[1].data

    fig = plt.gcf()
    plt.imshow(CUB200_loader.tensor_to_img(x1[0]), aspect='equal'), plt.axis('off'), fig.set_size_inches(448/100.0/3.0, 448/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
    plt.text(0, 0, f'L_cls = {cls_loss:.7f}, L_rank = {rank_loss:.7f}', color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
    plt.savefig(f'build/.cache/step{epoch}@loss={cls_loss+rank_loss}.jpg', dpi=300, pad_inches=0)    # visualize masked image

    fig = plt.gcf()
    plt.imshow(CUB200_loader.tensor_to_img(x2[0]), aspect='equal'), plt.axis('off'), fig.set_size_inches(448/100.0/3.0, 448/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
    plt.text(0, 0, f'L_cls = {cls_loss:.7f}, L_rank = {rank_loss:.7f}', color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
    plt.savefig(f'build/.cache/step{epoch}@loss={cls_loss+rank_loss}_4x.jpg', dpi=300, pad_inches=0)    # visualize masked image


def eval(net, dataloader):
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


def run():
    net = RACNN(num_classes=200).cuda()
    net.load_state_dict(torch.load('build/racnn_pretrained-1577262631.pt'))
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

    for epoch in range(260):
        cls_loss = train(net, trainloader, cls_opt, epoch, 'backbone', sample)
        rank_loss = train(net, trainloader, apn_opt, epoch, 'apn', sample)

        # eval_attention_sample(net, sample, cls_loss, rank_loss, epoch)
        # eval(net, testloader)
        if epoch > 2:
            return

        if epoch % 20 == 0 and epoch != 0:
            stamp = f'e{epoch}{int(time.time())}'
            torch.save(net, f'build/racnn_mobilenetv2_cub200-{stamp}.pt')
            torch.save(cls_opt.state_dict, f'build/cls_optimizer-{stamp}.pt')
            torch.save(apn_opt.state_dict, f'build/apn_optimizer-{stamp}.pt')


def clean(path='build/.cache/'):
    print(' :: Cleaning cache dir ...')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def build_gif(path='build/.cache'):
    files = os.listdir(path)
    files = [x for x in files if '4x' not in x]
    files.sort(key=lambda x: int(x.split('@')[0].split('epoch')[-1]))
    gif_images = []
    for img_file in files:
        gif_images.append(imageio.imread(f'{path}/{img_file}'))
    imageio.mimsave(f"build/racnn@2x-{int(time.time())}.gif", gif_images, fps=12)


if __name__ == "__main__":
    clean()
    run()
    build_gif()
