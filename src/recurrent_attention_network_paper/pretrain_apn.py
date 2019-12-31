import shutil
import cv2
import imageio
import os
import numpy as np
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
from torch.autograd import Variable



def log(msg):
    open('build/core.log', 'a').write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]\t'+msg+'\n'), print(msg)


def random_sample(dataloader):
    for batch_idx, (inputs, _) in enumerate(dataloader, 0):
        return inputs[0].cuda()


def save_img(x, path, annotation=''):
    fig = plt.gcf()  # generate outputs
    plt.imshow(CUB200_loader.tensor_to_img(x[0]), aspect='equal'), plt.axis('off'), fig.set_size_inches(448/100.0/3.0, 448/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
    plt.text(0, 0, annotation, color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
    plt.savefig(path, dpi=300, pad_inches=0)    # visualize masked image


def run(pretrained_backbone=None):
    net = RACNN(num_classes=200).cuda()
    if pretrained_backbone:  # Using pretrained backbone for apn pretraining
        state_dict = torch.load(pretrained_backbone).state_dict()
        net.b1.load_state_dict(state_dict)
        net.b2.load_state_dict(state_dict)
        net.b3.load_state_dict(state_dict)

    cudnn.benchmark = True

    params = list(net.apn1.parameters()) + list(net.apn2.parameters())
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

    trainset = CUB200_loader('external/CUB_200_2011', split='train')
    testset = CUB200_loader('external/CUB_200_2011', split='test')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=trainset.CUB_collate, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, collate_fn=testset.CUB_collate, num_workers=4)
    sample = random_sample(testloader)
    net.mode("pretrain_apn")

    def avg(x): return sum(x)/len(x)
    for epoch in range(1):
        losses = []
        for step, (inputs, _) in enumerate(trainloader, 0):

            loss = net.echo(inputs, optimizer)
            losses.append(loss)
            avg_loss = avg(losses[-5 if len(losses) > 5 else -len(losses):])
            print(f':: loss @step{step:2d}: {loss}\tavg_loss_5: {avg_loss}')

            if step % 2 == 0 or step < 5:  # check point
                _, _, _, resized = net(sample.unsqueeze(0))
                x1, x2 = resized[0].data, resized[1].data
                # visualize cropped inputs
                save_img(x1, path=f'build/.cache/step_{step}@2x.jpg', annotation=f'loss = {avg_loss:.7f}, step = {step}')
                save_img(x2, path=f'build/.cache/step_{step}@4x.jpg', annotation=f'loss = {avg_loss:.7f}, step = {step}')

            if step >= 128:  # 128 steps is enough for pretraining
                torch.save(net.state_dict(), f'build/racnn_pretrained-{int(time.time())}.pt')
                return


def build_gif(pattern='@2x', gif_name='pretrain_apn_cub200', cache_path='build/.cache'):
    # generate a gif, enjoy XD
    files = [x for x in os.listdir(cache_path) if pattern in x]
    files.sort(key=lambda x: int(x.split('@')[0].split('_')[-1]))
    gif_images = [imageio.imread(f'{cache_path}/{img_file}') for img_file in files]
    imageio.mimsave(f"build/{gif_name}{pattern}-{int(time.time())}.gif", gif_images, fps=8)


def clean(path='build/.cache/'):
    print(' :: Cleaning cache dir ...')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    clean()
    run(pretrained_backbone='build/mobilenet_v2_cub200-e801577256085.pt')
    build_gif(pattern='@2x', gif_name='pretrain_apn_cub200')
    build_gif(pattern='@4x', gif_name='pretrain_apn_cub200')
