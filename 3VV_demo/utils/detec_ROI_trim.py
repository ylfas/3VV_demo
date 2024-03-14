import os
import cv2
import numpy as np

def find_min_max_coordinates(detections):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for detection in detections:
        detection = detection.strip().split(' ')
        # 将坐标字符串转换为浮点数
        class_id, center_x, center_y, width, height, confidence = map(float, detection)

        # 计算类别框的最小和最大坐标值
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        # 更新最小和最大坐标值
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    return min_x, min_y, max_x, max_y


predict_txt_path = r'.../yolo_demo/runs/detect/exp4/labels/'

full_size_img_path = '.../yolo_demo/runs/detect/exp4/test/'

full_size_img_path_all = '.../data/test/img/'

for img in os.listdir(full_size_img_path):
    if img.split('.')[-1] == 'jpg':

        with open(predict_txt_path + 'test_' + img.replace('.jpg', '.txt'), 'r') as f:
            image = cv2.imread(full_size_img_path_all + img)
            mask = cv2.imread(full_size_img_path_all.replace('/img/', '/mask/') + img.replace('.jpg', '.png'))

            image_height, image_width = image.shape[:2]

            detections = f.readlines()
            # 找出最小和最大坐标值
            min_x, min_y, max_x, max_y = find_min_max_coordinates(detections)

            # 计算最小矩形框的左上角和右下角坐标
            left_top = [int(min_x * image_width), int(min_y * image_height)]
            right_bottom = [int(max_x * image_width), int(max_y * image_height)]

            if left_top[0] - 5 < 0:
                left_top[0] = 0
            else:
                left_top[0] = left_top[0] - 5

            if left_top[1] - 5 < 0:
                left_top[1] = 0
            else:
                left_top[1] = left_top[1] - 5

            if right_bottom[0] + 5 > image_width:
                right_bottom[0] = image_width
            else:
                right_bottom[0] = right_bottom[0] + 5

            if right_bottom[0] + 5 > image_height:
                right_bottom[0] = image_height
            else:
                right_bottom[0] = right_bottom[0] + 5

            #
            print("最小矩形框左上角坐标：", left_top)
            print("最小矩形框右下角坐标：", right_bottom)

            cropped_image = image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
            cropped_mask = mask[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]

            cv2.imwrite("/data/ylf/3VV_demo/result/new_dataset/test/img/" + img, cropped_image)
            cv2.imwrite("/data/ylf/3VV_demo/result/new_dataset/test/mask/" + img.replace('.jpg', '.png'), cropped_mask)


            with open('/data/ylf/3VV_demo/result/new_dataset/test/txt/' + img.replace('.jpg', '.txt'), 'w') as t:
                t.write(str(left_top[0]) + ' ' + str(left_top[1]) + ' ' + str(right_bottom[0]) + ' ' + str(right_bottom[1]))
            t.close()

        f.close()
