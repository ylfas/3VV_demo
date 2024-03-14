import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from numpy import random
from torchvision import transforms
from torchvision import utils as vutils
from dataset import *
from torchvision.datasets import ImageFolder
import numpy as np
import os
import cv2

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mean_and_std(dataset_dir):
    images = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                images.append(os.path.join(root, file))

    pixel_num = 0
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)

    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
        img = img.astype(np.float32) / 255.

        pixel_num += (img.size / 3)
        channel_sum += np.sum(img, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(img), axis=(0, 1))

    channel_mean = channel_sum / pixel_num
    channel_std = np.sqrt(channel_sum_squared / pixel_num - np.square(channel_mean))

    return channel_mean, channel_std

if __name__ == '__main__':

    #离线数据增强
    images_path = r"/t9k/mnt/data/BUSI/train/img/"
    masks_path = images_path.replace('/img/', '/mask/')

    count = 0
    for img in os.listdir(images_path):
        if img.split('.')[-1] == 'jpg':
            count += 1
            image = Image.open(images_path + img)
            mask = Image.open(masks_path + img.replace('jpg', 'png'))

            num = random.choice([1, 3, 5, 7, 9])
            # set_seed(seed)
            seed = np.random.randint(123456789)
            # img_x = self.transform(image)
            if num == 1:
                x_transforms = transforms.RandomHorizontalFlip(0.5)
                set_seed(seed)
                img_x = x_transforms(image)
                set_seed(seed)
                img_y = x_transforms(mask)

            if num == 3:
                x_transforms = transforms.RandomRotation(25)
                set_seed(seed)
                img_x = x_transforms(image)
                set_seed(seed)
                img_y = x_transforms(mask)

            if num == 5:
                x_transforms = transforms.RandomVerticalFlip(0.5)
                set_seed(seed)
                img_x = x_transforms(image)
                set_seed(seed)
                img_y = x_transforms(mask)

            if num == 7:
                x_transforms = transforms.Resize((525, 350), interpolation=2)
                set_seed(seed)
                img_x = x_transforms(image)
                set_seed(seed)
                img_y = x_transforms(mask)

            if num == 9:
                x_transforms = transforms.RandomRotation(35)
                set_seed(seed)
                img_x = x_transforms(image)
                set_seed(seed)
                img_y = x_transforms(mask)

            img_x.save(r"/t9k/mnt/data/BUSI/train/img_argu/" + img.replace('.', '_3.'))
            img_y.save(r"/t9k/mnt/data/BUSI/train/mask_argu/" + img.replace('.jpg', '_3.png'))

    print(count)



