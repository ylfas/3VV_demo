import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from utils1 import *
from helpers import *
from skimage.io import imread
import imageio
class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

def get_hd_2cls(mask_name,predict):

    image_mask1 = keep_image_size_open(mask_name)
    image_mask = np.array(image_mask1)
    # image_mask = cv2.imread(mask_name, 0)
    # print(mask_name)
    # print(image_mask)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))

    image_mask = np.where(image_mask < 0.5, 0, 1)

    hd1 = directed_hausdorff(image_mask, predict)[0]
    hd2 = directed_hausdorff(predict, image_mask)[0]
    res = 0

    if hd1 > hd2 or hd1 == hd2:
        res = hd1
    else:
        res = hd2

    return res

def get_iou_2cls(mask_name, predict):
    image_mask1 = keep_image_size_open(mask_name)
    image_mask = np.array(image_mask1)
    # image_mask = cv2.imread(mask_name,0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))

    image_mask = np.where(image_mask < 0.5, 0, 1)
    predict = predict.astype(np.int16)

    iou_tem = []

    interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    iou_tem.append(inter / union)

    intersection = []
    dice = []


    intersection.append((predict*image_mask).sum())
    dice.append((2. *intersection[0]) /(predict.sum()+image_mask.sum()))

    print('%s:Mean_dice=%f' % (mask_name, dice[0]))
    # print('%s:P_dice=%f' % (mask_name, dice))

    return iou_tem, dice

def get_hd_ce(mask_name,predict):

    image_mask1 = keep_image_size_open(mask_name)
    image_mask = np.array(image_mask1)
    # image_mask = cv2.imread(mask_name, 0)

    image_mask = np.expand_dims(image_mask, axis=2)
    image_mask = mask_to_onehot_three(mask_name, image_mask)
    image_mask = image_mask.transpose([2, 0, 1])
    #image_mask = mask

    hd1 = directed_hausdorff(1-image_mask[0], 1-predict[0])[0]
    hd2 = directed_hausdorff(1-predict[0], 1-image_mask[0])[0]
    res = 0

    hd_p1 = directed_hausdorff(image_mask[1], predict[1])[0]
    hd_p2 = directed_hausdorff(predict[1], image_mask[1])[0]
    res_p = 0

    hd_a1 = directed_hausdorff(image_mask[2], predict[2])[0]
    hd_a2 = directed_hausdorff(predict[2], image_mask[2])[0]
    res_a = 0

    hd_v1 = directed_hausdorff(image_mask[3], predict[3])[0]
    hd_v2 = directed_hausdorff(predict[3], image_mask[3])[0]
    res_v = 0

    if hd1 > hd2:
        res = hd1
    else:
        res = hd2

    if hd_p1 > hd_p2:
        res_p = hd_p1
    else:
        res_p = hd_p2

    if hd_a1 > hd_a2:
        res_a = hd_a1
    else:
        res_a = hd_a2

    if hd_v1 > hd_v2:
        res_v = hd_v1
    else:
        res_v = hd_v2

    return res, res_p, res_a, res_v

def get_iou_ce(mask_name, predict):
    image_mask1 = keep_image_size_open(mask_name)
    image_mask = np.array(image_mask1)
    # image_mask = cv2.imread(mask_name,0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))

    image_mask = np.expand_dims(image_mask, axis=2)
    image_mask = mask_to_onehot_three(mask_name, image_mask)
    image_mask = image_mask.transpose([2, 0, 1])
    # image_mask = mask

    iou_tem = []
    for i in range(3):
        interArea = np.multiply(predict[i+1], image_mask[i+1])
        tem = predict[i+1] + image_mask[i+1]
        unionArea = tem - interArea
        inter = np.sum(interArea)
        union = np.sum(unionArea)
        iou_tem.append(inter / union)

    # print('%s:P_iou=%f, A_iou=%f, V_iou=%f' % (mask_name, iou_tem[0], iou_tem[1], iou_tem[2]))

    intersection = []
    dice = []

    for i in range(3):
        intersection.append((predict[i+1]*image_mask[i+1]).sum())
        dice.append((2. *intersection[i]) /(predict[i+1].sum()+image_mask[i+1].sum()))

    return iou_tem, dice


def get_hd_3rtim(mask_name, predict):

    image_mask1 = keep_image_size_open(mask_name)
    image_mask = np.array(image_mask1)

    image_mask = np.expand_dims(image_mask, axis=2)
    image_mask = mask_to_onehot_three(mask_name, image_mask)
    image_mask = image_mask.transpose([2, 0, 1])
    #image_mask = mask
    predict = predict.transpose([2, 0, 1])

    hd_p1 = directed_hausdorff(image_mask[1], predict[0])[0]
    hd_p2 = directed_hausdorff(predict[0], image_mask[1])[0]
    res_p = 0

    hd_a1 = directed_hausdorff(image_mask[2], predict[1])[0]
    hd_a2 = directed_hausdorff(predict[1], image_mask[2])[0]
    res_a = 0

    hd_v1 = directed_hausdorff(image_mask[3], predict[2])[0]
    hd_v2 = directed_hausdorff(predict[2], image_mask[3])[0]
    res_v = 0


    if hd_p1 > hd_p2:
        res_p = hd_p1
    else:
        res_p = hd_p2

    if hd_a1 > hd_a2:
        res_a = hd_a1
    else:
        res_a = hd_a2

    if hd_v1 > hd_v2:
        res_v = hd_v1
    else:
        res_v = hd_v2

    return res_p, res_a, res_v

def get_iou_3trim(mask_name, predict):
    image_mask1 = keep_image_size_open(mask_name)
    image_mask = np.array(image_mask1)
    # image_mask = cv2.imread(mask_name,0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))

    image_mask = np.expand_dims(image_mask, axis=2)
    image_mask = mask_to_onehot_three(mask_name, image_mask)
    image_mask = image_mask.transpose([2, 0, 1])

    predict = predict.transpose([2, 0, 1])

    iou_tem = []
    for i in range(3):
        interArea = np.multiply(predict[i], image_mask[i+1])
        tem = predict[i] + image_mask[i+1]
        unionArea = tem - interArea
        inter = np.sum(interArea)
        union = np.sum(unionArea)
        iou_tem.append(inter / union)

    print('%s:P_iou=%f, A_iou=%f, V_iou=%f' % (mask_name, iou_tem[0], iou_tem[1], iou_tem[2]))

    intersection = []
    dice = []

    for i in range(3):
        intersection.append((predict[i]*image_mask[i+1]).sum())
        dice.append((2. *intersection[i]) /(predict[i].sum()+image_mask[i+1].sum()))

    print('%s:P_dice=%f, A_dice=%f, V_dice=%f' % (mask_name, dice[0], dice[1], dice[2]))
    # print('%s:P_dice=%f' % (mask_name, dice))

    return iou_tem, dice


def show(predict):
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            predict[row, col] *= 255
    plt.imshow(predict)
    plt.show()

