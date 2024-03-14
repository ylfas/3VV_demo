import os
import json
import  numpy as np
from PIL import Image

# class name
classes = ['0', '1']
# 初始化二维0数组
result_list = np.array(np.zeros([len(classes), len(classes)]))


def convert(img_size, box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    # 转换并归一化
    center_x = (x1 + x2) * 0.5 / img_size[0]
    center_y = (y1 + y2) * 0.5 / img_size[1]
    w = abs((x2 - x1)) * 1.0 / img_size[0]
    h = abs((y2 - y1)) * 1.0 / img_size[1]

    return (round(center_x, 5), round(center_y, 5), round(w, 5), round(h, 5))


def convert_annotation(image_id, json_txt):
    # in_file = open('/home/zhyl_ylf/dataset/data/Annotations/%s.json' % (image_id), encoding='UTF-8')


    global bbox

    json_path = image_id
    data = json.load(open(json_path, 'r'))
    img_w = data['imageWidth']
    img_h = data["imageHeight"]  # te files

    for i in data['shapes']:
        j = i.get("label")
        if i['points']:  # 仅适用矩形框标注
            i = i['points']
            x1 = max(float(i[0][0]), 0)

            y1 = max(float(i[0][1]), 0)

            x2 = max(float(i[1][0]), 0)

            y2 = max(float(i[1][1]), 0)

            bb = (x1, y1, x2, y2)

            bbox = convert((img_w, img_h), bb)

        # if json_file.split('_')[0] == 'benign':
        #     cls = 0  # 得到当前label的类别
        # else:
        #     cls = 1

            # 转换成训练模式读取的标签
            # cls_id = classes.index(cls)  # 位于定义类别索引位置

            # 保存
    json_txt.write(str(0) + ' ' + " ".join([str(a) for a in bbox]) + "\n")  # 生成格式0 cx,cy,w,h

# 获取图片宽高
def get_image_width_high(full_image_name):
    image = Image.open(full_image_name)
    image_width, image_high = image.size[0], image.size[1]
    return image_width, image_high


# 读取原始标注数据
def read_label_txt(full_label_name, full_image_name):
    fp = open(full_label_name, mode="r")
    lines = fp.readlines()
    image_width, image_high = get_image_width_high(full_image_name)
    object_list = []
    for line in lines:
        array = line.split()
        x_label_min = (float(array[1]) - float(array[3]) / 2) * image_width
        x_label_max = (float(array[1]) + float(array[3]) / 2) * image_width
        y_label_min = (float(array[2]) - float(array[4]) / 2) * image_high
        y_label_max = (float(array[2]) + float(array[4]) / 2) * image_high
        bbox = [round(x_label_min, 2), round(y_label_min, 2), round(x_label_max, 2), round(y_label_max, 2)]
        category = int(array[0])
        obj_info = {
            'category' : category,
            'bbox' : bbox
        }
        object_list.append(obj_info)
    return object_list


# 计算交集面积
def label_area_detect(label_bbox_list, detect_bbox_list):
    x_label_min, y_label_min, x_label_max, y_label_max = label_bbox_list
    x_detect_min, y_detect_min, x_detect_max, y_detect_max = detect_bbox_list
    if (x_label_max <= x_detect_min or x_detect_max < x_label_min) or ( y_label_max <= y_detect_min or y_detect_max <= y_label_min):
        return 0
    else:
        lens = min(x_label_max, x_detect_max) - max(x_label_min, x_detect_min)
        wide = min(y_label_max, y_detect_max) - max(y_label_min, y_detect_min)
        return lens * wide

# label 匹配 detect
def label_match_detect(image_name, label_list, detect_list,full_image_name):
    for label in label_list:
        area_max = 0
        area_category = 0
        detect_bbox1 = []
        label_category = label['category']
        label_bbox = label['bbox']
        for detect in detect_list:
            if detect.replace('.txt','.png') == image_name:
                with open('/data/ylf/yolov5/yolov5-5.0/runs/detect/exp31/labels/' + detect, 'r') as f:
                    detect_bbox = f.readlines()
                    detect_bbox = str(detect_bbox).split()
                    # detect_bbox =
                f.close()
                for i in range(4):
                    detect_bbox1.append(detect_bbox[i+1])
                area_category = detect_bbox[0].split("['")[1]

                image_width, image_high = get_image_width_high(full_image_name)
                x_label_min = (float(detect_bbox1[0]) - float(detect_bbox1[2]) / 2) * image_width
                x_label_max = (float(detect_bbox1[0]) + float(detect_bbox1[2]) / 2) * image_width
                y_label_min = (float(detect_bbox1[1]) - float(detect_bbox1[3]) / 2) * image_high
                y_label_max = (float(detect_bbox1[1]) + float(detect_bbox1[3]) / 2) * image_high

                bbox = [round(x_label_min, 2), round(y_label_min, 2), round(x_label_max, 2), round(y_label_max, 2)]
                area = label_area_detect(label_bbox, bbox)
                if area > area_max:
                    area_max = area
                    # area_category = detect['category']

        # result_list[int(label_category)][classes.index(str(area_category))] += 1
        result_list[int(label_category)][classes.index(str(area_category))] += 1


def main():
    image_path = '/data/ylf/yolov5/K9RDPP9G_m.avi/'  # 图片文件路径
    label_path = '/data/ylf/yolov5/K9RDPP9G_m.avi/'  # 标注文件路径
    detect_path = '/data/ylf/yolov5/yolov5-5.0/runs/detect/exp31/labels/'    # 预测的数据
    detect_list = os.listdir(detect_path)
    precision = 0     # 精确率
    recall = 0        # 召回率
    # 读取 预测 文件数据
    # with open(detect_path, 'r') as load_f:
    #     detect_list = json.load(load_f)

    # txts_path = os.listdir(detect_path)
    # lines = []
    # for txt in txts_path:
    #     with open(detect_path + txt, 'r') as f:
    #         # lines.append(str(txt))
    #         lines.append(f.readlines())
    #     f.close()

    # 读取图片文件数据
    all_files = os.listdir(image_path)
    img_nums = 0
    imgs = []
    json_nums = 0
    jsons = []
    for file in all_files:
        if file.split('.')[-1] == 'png':
            imgs.append(file)
        if file.split('.')[-1] == 'json':
            jsons.append(file)
    all_image = imgs

    for i in range(len(all_image)):
        full_image_path = os.path.join(image_path, all_image[i])
        # 分离文件名和文件后缀
        image_name, image_extension = os.path.splitext(all_image[i])
        # 拼接标注路径
        full_label_path = os.path.join(label_path, image_name + '.json')
        # if os.path.exists(full_label_path):
        #
        #     json_txt = open(full_label_path.split('.json')[0] + '.txt', 'w')
        #     convert_annotation(full_label_path, json_txt)
        #     json_txt.close()
        # else:
        #     json_txt = open(full_label_path.split('.json')[0] + '.txt', 'w')
        #     json_txt.write('1' + ' 0 0 0 0')
        #     json_txt.close()

            # 读取标注数据
        label_list = read_label_txt(full_label_path.replace('.json','.txt'), full_image_path)

            # 标注数据匹配detect
        label_match_detect(all_image[i], label_list, detect_list, full_image_path)
    print(result_list)
    for i in range(len(classes)):
        row_sum, col_sum = sum(result_list[i]), sum(result_list[r][i] for r in range(len(classes)))
        precision += result_list[i][i] / float(col_sum)
        recall += result_list[i][i] / float(row_sum)
    print(f'precision: {precision / len(classes) * 100}%  recall: {recall / len(classes) * 100}%')

    print(f'with_Nodule     precision: {result_list[0][0]/(result_list[0][0]+result_list[1][0])}      recall: {result_list[0][0]/(result_list[0][0]+result_list[0][1])}')

    print(f'without_Nodule  precision: {result_list[1][1] / (result_list[1][1] + result_list[0][1])}  recall: {result_list[1][1] / (result_list[1][1] + result_list[1][0])}')


if __name__ == '__main__':
    main()