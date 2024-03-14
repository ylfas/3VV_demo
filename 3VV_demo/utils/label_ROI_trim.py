import os
import cv2
import numpy as np

full_size_img_path = '.../data/val/img/'

for img in os.listdir(full_size_img_path):
    if img.split('.')[-1] == 'jpg':

        image = cv2.imread(full_size_img_path + img)
        # mask = cv2.imread(full_size_img_path.replace('/img/', '/mask/') + img.replace('.jpg', '.png'))
        image_height, image_width = image.shape[:2]

        # 读取mask矩阵
        mask = cv2.imread(full_size_img_path.replace('/img/', '/mask/') + img.replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化最小外接矩形列表
        min_rectangles = []
        # 计算所有目标的最小外接矩形的包围框
        all_rectangles = [cv2.boundingRect(contour) for contour in contours]

        # 将边界框坐标转换为NumPy数组
        all_rectangles = np.array(all_rectangles)

        # 计算所有边界框的左上角和右下角坐标
        min_x = np.min(all_rectangles[:, 0])
        min_y = np.min(all_rectangles[:, 1])
        max_x = np.max(all_rectangles[:, 0] + all_rectangles[:, 2])
        max_y = np.max(all_rectangles[:, 1] + all_rectangles[:, 3])

        if min_x - 5 < 0:
            min_x = 0
        else:
            min_x = min_x - 5

        if min_y - 5 < 0:
            min_y = 0
        else:
            min_y = min_y - 5

        if max_x + 5 > image_width:
            max_x = image_width
        else:
            max_x = max_x + 5

        if max_y + 5 > image_height:
            max_y = image_height
        else:
            max_y = max_y + 5

        cropped_image = image[min_y:max_y, min_x:max_x]
        cropped_mask = mask[min_y:max_y, min_x:max_x]

        cv2.imwrite(".../3VV_demo/result/new_dataset/val/img/" + img, cropped_image)
        cv2.imwrite(".../3VV_demo/result/new_dataset/val/mask/" + img.replace('.jpg', '.png'), cropped_mask)

