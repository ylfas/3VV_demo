import torch
import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
from torchvision.transforms import transforms
# from main import set_seed
from utils1 import *
from helpers import *
import matplotlib.pyplot as plt

palette = [[85], [170], [255]]
palette_ce = [[0], [85], [170], [255]]
palette_BUSI = [[0], [255]]
num_classes = 3

class MyDataset_full_size(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):

        self.state = state
        self.train_root = r"/data/ylf/3VV_test/fold1/train/"
        self.val_root = r"/data/ylf/3VV_test/fold1/val/"
        self.test_root = r'/data/ylf/3VV_test/fold1/test/'
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform
        self.normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.tensor = transforms.ToTensor()

    def getDataPath(self):
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        # n = len(os.listdir(root + 'mask/'))  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        I = os.listdir(root + 'mask/')
        for i in I:
            if i.split('.')[-1] in ['jpg', 'png']:
                img = os.path.join(root + 'img/', i.replace('png', 'jpg'))
                # img = os.path.join(root + 'img/', i)# liver is %03d
                mask = os.path.join(root + 'mask/', i)
                pics.append(img)
                masks.append(mask)
                # imgs.append((img, mask))
        return pics, masks

    def __getitem__(self, index):
        # x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]

        # img_tem = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_tem, connectivity=8)

        segment_image = keep_image_size_open(y_path)
        image = keep_image_size_open(x_path)

        img = np.array(image)
        mask = np.array(segment_image)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
        # img = np.expand_dims(img, axis=2)

        mask = np.expand_dims(mask, axis=2)
        mask = mask_to_onehot(y_path, mask, [85, 170, 255])

        if self.transform is not None:
            # set_seed(seed)
            # img_x = self.transform(image)
            img_x = self.tensor(img)
            img_x = self.normal(img_x)

            img_y = self.tensor(mask)
        return img_x, img_y, x_path, y_path

    def __len__(self):
        return len(self.pics)


class MyDataset_CE(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):

        self.state = state
        self.train_root = r"/data/ylf/3VV_test/fold1/train/"
        self.val_root = r"/data/ylf/3VV_test/fold1/val/"
        self.test_root = r'/data/ylf/3VV_test/fold1/test/'
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform
        self.normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.tensor = transforms.ToTensor()

    def getDataPath(self):
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        # n = len(os.listdir(root + 'mask/'))  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        I = os.listdir(root + 'mask/')
        for i in I:
            img = os.path.join(root + 'img/', i.replace('png', 'jpg'))
            # img = os.path.join(root + 'img/', i)# liver is %03d
            mask = os.path.join(root + 'mask/', i)
            pics.append(img)
            masks.append(mask)
            # imgs.append((img, mask))
        return pics, masks

    def __getitem__(self, index):
        # x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]

        # img_tem = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_tem, connectivity=8)

        segment_image = keep_image_size_open(y_path)
        image = keep_image_size_open(x_path)

        img = np.array(image)
        mask = np.array(segment_image)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
        # img = np.expand_dims(img, axis=2)

        mask_onehot = np.expand_dims(mask, axis=2)
        mask_onehot = mask_to_onehot('', mask_onehot, palette_ce)

        # img = np.array(image)
        # mask = np.array(segment_image)
        mask_label = np.expand_dims(mask, axis=2)
        mask_label = mask_to_onehot_three(palette, mask_label)
        # mask = Image.fromarray(np.uint8(mask))
        # #
        # seed = np.random.randint(123456789)
        # set_seed(seed)

        if self.transform is not None:
            # set_seed(seed)
            # img_x = self.transform(image)
            img_x = self.tensor(img)
            img_x = self.normal(img_x)

            ce_label = torch.zeros_like(img_x[0], dtype=torch.long)
            ce_label[mask == 85] = 1
            ce_label[mask == 170] = 2
            ce_label[mask == 255] = 3

            img_y = self.tensor(mask_label)
        return img_x, img_y, x_path, y_path, ce_label

    def __len__(self):
        return len(self.pics)


class MyDataset_trim(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):

        self.state = state
        self.train_root = r"/data/ylf/3VV_demo/result/new_dataset/train/"
        self.val_root = r"/data/ylf/3VV_demo/result/new_dataset/val/"
        self.test_root = r'/data/ylf/3VV_demo/result/new_dataset/test/'
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform
        self.normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.tensor = transforms.ToTensor()

    def getDataPath(self):
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        # n = len(os.listdir(root + 'mask/'))  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        I = os.listdir(root + 'mask/')
        for i in I:
            img = os.path.join(root + 'img/', i.replace('png', 'jpg'))
            # img = os.path.join(root + 'img/', i)# liver is %03d
            mask = os.path.join(root + 'mask/', i)
            pics.append(img)
            masks.append(mask)
            # imgs.append((img, mask))
        return pics, masks

    def __getitem__(self, index):
        # x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]

        # img_tem = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_tem, connectivity=8)

        segment_image = keep_image_size_open(y_path)
        image = keep_image_size_open(x_path)

        img = np.array(image)
        mask = np.array(segment_image)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
        # img = np.expand_dims(img, axis=2)

        mask_onehot = np.expand_dims(mask, axis=2)
        mask_onehot = mask_to_onehot('', mask_onehot, palette_ce)

        # img = np.array(image)
        # mask = np.array(segment_image)
        mask_label = np.expand_dims(mask, axis=2)
        mask_label = mask_to_onehot_three(palette, mask_label)
        # mask = Image.fromarray(np.uint8(mask))
        # #
        # seed = np.random.randint(123456789)
        # set_seed(seed)

        if self.transform is not None:
            # set_seed(seed)
            # img_x = self.transform(image)
            img_x = self.tensor(img)
            img_x = self.normal(img_x)

            ce_label = torch.zeros_like(img_x[0], dtype=torch.long)
            ce_label[mask == 85] = 1
            ce_label[mask == 170] = 2
            ce_label[mask == 255] = 3

            img_y = self.tensor(mask_label)
        return img_x, img_y, x_path, y_path, ce_label

    def __len__(self):
        return len(self.pics)


class MyDataset_2cls(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):

        self.state = state
        self.train_root = r"/data/ylf/3VV_test/fold1/train/"
        self.val_root = r"/data/ylf/3VV_test/fold1/val/"
        self.test_root = r'/data/ylf/3VV_test/fold1/test/'
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform
        self.normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.tensor = transforms.ToTensor()

    def getDataPath(self):
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        # n = len(os.listdir(root + 'mask/'))  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        I = os.listdir(root + 'mask/')
        for i in I:
            if i.split('.')[-1] == 'png':
                img = os.path.join(root + 'img/', i.replace('png', 'jpg'))
                # img = os.path.join(root + 'img/', i)# liver is %03d
                mask = os.path.join(root + 'mask/', i)
                pics.append(img)
                masks.append(mask)
                # imgs.append((img, mask))
        return pics, masks

    def __getitem__(self, index):
        # x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]

        # img_tem = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_tem, connectivity=8)

        segment_image = keep_image_size_open(y_path)
        image = keep_image_size_open(x_path)

        img = np.array(image)
        mask = np.array(segment_image)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
        # img = np.expand_dims(img, axis=2)


        mask_onehot = np.where(mask>2, 1, 0)


        if self.transform is not None:
            # set_seed(seed)
            # img_x = self.transform(image)
            img_x = self.tensor(img)
            img_x = self.normal(img_x)

            # ce_label = torch.zeros_like(img_x[0], dtype=torch.float32)
            # ce_label[mask == 85] = 1
            # ce_label[mask == 170] = 2
            # ce_label[mask == 255] = 3

            img_y = self.tensor(mask_onehot)
            img_y = img_y.to(torch.float32)
        return img_x, img_y, x_path, y_path

    def __len__(self):
        return len(self.pics)





