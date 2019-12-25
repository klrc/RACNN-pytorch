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


def run():
    net = RACNN(num_classes=200).cuda()
    cudnn.benchmark = True

    params = list(net.apn1.parameters()) + list(net.apn2.parameters())
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)

    trainset = CUB200_loader('external/CUB_200_2011', split='train')
    testset = CUB200_loader('external/CUB_200_2011', split='test')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=trainset.CUB_collate, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, collate_fn=testset.CUB_collate, num_workers=4)
    sample = random_sample(testloader)
    net.mode("pretrain_apn")

    for epoch in range(1):
        def avg(x): return sum(x)/len(x)
        losses = []
        for step, (inputs, _) in enumerate(trainloader, 0):

            loss = net.echo(inputs, optimizer)
            losses.append(loss)
            avg_loss = avg(losses[-5 if len(losses) > 5 else -len(losses):])
            print(f':: loss @step{step:2d}: {loss}\tavg_loss_5: {avg_loss}')

            if step % 2 == 0 or step < 5:  # check point
                _, _, attens, resized = net(sample.unsqueeze(0))
                x1, x2 = resized[0].data, resized[1].data

                fig = plt.gcf()
                plt.imshow(CUB200_loader.tensor_to_img(x1[0]), aspect='equal'), plt.axis('off'), fig.set_size_inches(448/100.0/3.0, 448/100.0/3.0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
                plt.text(0, 0, f'loss = {avg_loss:.7f}, step = {step}', color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
                plt.savefig(f'build/.cache/step{step}@loss={avg_loss}.jpg', dpi=300, pad_inches=0)    # visualize masked image
            if step >= 64:
                torch.save(net.state_dict(), 'build/racnn_pretrained.pt')
                return


def build_gif(path='build/.cache'):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x.split('@')[0].split('step')[-1]))
    gif_images = []
    for img_file in files:
        gif_images.append(imageio.imread(f'{path}/{img_file}'))
    imageio.mimsave(f"build/pretrain_apn.gif", gif_images, fps=6)

if __name__ == "__main__":    
    run()
    build_gif()
