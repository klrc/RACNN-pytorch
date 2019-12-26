import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class CUB200_loader(data.Dataset):
    def __init__(self, root, split='train', transform=None):

        std = 1. / 255.
        means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]

        split_list = open(os.path.join(root, 'train_test_split.txt')).readlines()
        self.idx2name = []
        classes = open(os.path.join(root, 'classes.txt')).readlines()
        self._imgpath = []
        self._imglabel = []
        self.transform = None

        for c in classes:
            idx, name = c.strip().split()
            self.idx2name.append(name)

        if transform is None and split.lower() == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(448),
                transforms.RandomCrop([448, 448]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=means,
                    std=[std]*3)
            ])
        elif transform is None and split.lower() == 'test':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(448),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=means,
                    std=[std]*3)
            ])
        else:
            print(" [*] Warning: transform is not None, Recomend to use default")
            pass

        train_list = []
        test_list = []
        for line in split_list:
            idx, is_train = line.strip().split()
            if int(is_train) == 1:
                train_list.append(int(idx) - 1)
            else:
                test_list.append(int(idx) - 1)

        image_list = open(os.path.join(root, 'images.txt')).readlines()
        image_list = np.array(image_list)
        label_list = open(os.path.join(root, 'image_class_labels.txt')).readlines()
        label_list = np.array(label_list)

        if split.lower() == 'train':
            train_images = image_list[train_list]
            train_labels = label_list[train_list]
            for i, fname in enumerate(train_images):
                idx, path = fname.strip().split()
                self._imgpath.append(os.path.join(root, 'images', path))
                idx, label = train_labels[i].strip().split()
                self._imglabel.append(int(label) - 1)
        else:
            test_images = image_list[test_list]
            test_labels = label_list[test_list]
            for i, fname in enumerate(test_images):
                idx, path = fname.strip().split()
                self._imgpath.append(os.path.join(root, 'images', path))
                idx, label = test_labels[i].strip().split()
                self._imglabel.append(int(label) - 1)

    def __getitem__(self, index):
        img = cv2.imread(self._imgpath[index])
        img = self.transform(img)
        cls = self._imglabel[index]
        return img, cls

    def __len__(self):
        return len(self._imglabel)

    @staticmethod
    def tensor_to_img(x, imtype=np.uint8):
        """"将tensor的数据类型转成numpy类型，并反归一化.

        Parameters:
            input_image (tensor) --  输入的图像tensor数组
            imtype (type)        --  转换后的numpy的数据类型
        """
        mean = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
        std = [1. / 255., 1. / 255., 1. / 255.]

        if not isinstance(x, np.ndarray):
            if isinstance(x, torch.Tensor):  # get the data from a variable
                image_tensor = x.data
            else:
                return x
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            for i in range(len(mean)):
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
            image_numpy = image_numpy * 255
            image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
        else:  # if it is a numpy array, do nothing
            image_numpy = x
        return image_numpy.astype(imtype)

    def idx_to_classname(self, idx):
        return self.idx2name[idx]

    def CUB_collate(self, batch):
        imgs = []
        cls = []
        for sample in batch:
            imgs.append(sample[0])
            cls.append(sample[1])
        imgs = torch.stack(imgs, 0)
        cls = torch.LongTensor(cls)
        return imgs, cls


if __name__ == '__main__':
    trainset = CUB200_loader('external/CUB_200_2011')
    trainloader = data.DataLoader(trainset, batch_size=32,
                                  shuffle=False, collate_fn=trainset.CUB_collate, num_workers=1)
    for img, cls in trainloader:
        print(' [*] train images:', img.size())
        print(' [*] train class:', cls.size())
        break

    testset = CUB200_loader('external/CUB_200_2011')
    testloader = data.DataLoader(testset, batch_size=32,
                                 shuffle=False, collate_fn=testset.CUB_collate, num_workers=1)

    for img, cls in trainloader:
        print(' [*] test images:', img.size())
        print(' [*] test class:', cls.size())
        break
