import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from utils1 import *
from metrics import get_iou_3trim, get_hd_3rtim
import torchvision.transforms as transforms

mask_path = '/data/ylf/data/test/mask/'
predict_path = '/data/ylf/3VV_demo/result/2th_stage_seg/deeplabv3/'

resize = transforms.Resize((256, 256))

hd_total = 0
num = 0
hd_total_P, hd_total_A, hd_total_V = 0, 0, 0
miou_total_0, miou_total_1, miou_total_2 = 0, 0, 0
dice_total_0, dice_total_1, dice_total_2 = 0, 0, 0

for image in os.listdir(predict_path):
    num += 1

    img = cv2.imread(predict_path + image)
    pred = Image.fromarray(img)
    temp = max(pred.size)

    predict = Image.new('RGB', (temp, temp), (0, 0, 0))
    predict.paste(pred, (0, 0))
    predict = predict.resize((256, 256))

    predict = np.array(predict)

    predict = np.where(predict < 0.5, 0, 1)

    iou, dice = get_iou_3trim(mask_path + image, predict)
    hd = get_hd_3rtim(mask_path + image, predict)

    hd_total += hd[0] + hd[1] + hd[2]

    miou_total_0 += iou[0]  # 获取当前预测图的miou，并加到总miou中
    dice_total_0 += dice[0]
    miou_total_1 += iou[1]
    dice_total_1 += dice[1]
    miou_total_2 += iou[2]
    dice_total_2 += dice[2]

    hd_total_P += hd[0]
    hd_total_A += hd[1]
    hd_total_V += hd[2]

miou_total = miou_total_0 + miou_total_1 + miou_total_2
dice_total = dice_total_0 + dice_total_1 + dice_total_2

print('Miou=%f,aver_hd=%f,M_dice=%f' % (miou_total / (3 * num), hd_total / (3 * num), dice_total / (3 * num)))
print('P_iou=%f,A_iou=%f,V_iou=%f ,P_dice=%f,A_dice=%f,V_dice=%f' % (
miou_total_0 / num, miou_total_1 / num, miou_total_2 / num,
dice_total_0 / num, dice_total_1 / num, dice_total_2 / num))
print('P_hd = %f, A_hd = %f, V_hd = %f' % (hd_total_P / num, hd_total_A / num, hd_total_V / num))