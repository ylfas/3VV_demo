from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

def keep_image_size_open(path, size=(256, 256)):

    img = Image.open(path)
    # resize = transforms.Resize((256, 256))
    temp = max(img.size)
    if path.split('.')[-1] == 'png':
        I = img
        # I.show()
        L = I.convert('L')
        # L.show()
        img = L
        mask = Image.new('L', (temp, temp), 0)
        mask.paste(img, (0, 0))
        mask = mask.resize(size)
    else:
        mask = Image.new('RGB', (temp, temp), (0, 0, 0))
        mask.paste(img, (0, 0))
        mask = mask.resize(size)

    return mask


def recover_image_size(predict_image, mask_path):

    img = Image.open(mask_path)
    temp_w = img.size[0]
    temp_h = img.size[1]

    if temp_h > temp_w:
        rate = (temp_h - temp_w)/temp_h
        pis = int(256*(1-rate))
        # pis = int(pis * (temp_h/256))
        predict_image = Image.fromarray(np.uint8(predict_image[1:].transpose([1, 2, 0])))
        # predict_image = Image.fromarray((np.uint8(predict_image)).transpose([1, 2, 0]))
        cropped_1 = predict_image.crop((0, 0, pis, 256))  # (left, upper, right, lower)
        cropped_2 = np.array(cropped_1)
        # cropped = np.resize(cropped_2,(temp_h, temp_w))
        resized_image = cv2.resize(cropped_2, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)

        resized_image[:, :, 0] *= 85
        resized_image[:, :, 1] *= 170
        resized_image[:, :, 2] *= 255

        return resized_image

    else:

        rate = (temp_w - temp_h) / temp_w
        pis = int(256 * (1 - rate))
        # pis = int(pis * (temp_h/256))
        predict_image = Image.fromarray((np.uint8(predict_image[1:])).transpose([1, 2, 0]))
        cropped_1 = predict_image.crop((0, 0, 256, pis))  # (left, upper, right, lower)
        cropped_2 = np.array(cropped_1)
        resized_image = cv2.resize(cropped_2, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)

        resized_image[:, :, 0] *= 85
        resized_image[:, :, 1] *= 170
        resized_image[:, :, 2] *= 255
        # resized_image[:, :, 0] * 85 + resized_image[:, :, 1] * 170 + resized_image[:, :, 2] * 255

        return resized_image


def recover_image_size_1cls(predict_image, mask_path):

    img = Image.open(mask_path)
    temp_w = img.size[0]
    temp_h = img.size[1]

    if temp_h > temp_w:
        rate = (temp_h - temp_w)/temp_h
        pis = int(256*(1-rate))
        # pis = int(pis * (temp_h/256))
        predict_image = Image.fromarray(np.uint8(predict_image))
        cropped_1 = predict_image.crop((0, 0, pis, 256))  # (left, upper, right, lower)
        cropped_2 = np.array(cropped_1)
        # cropped = np.resize(cropped_2,(temp_h, temp_w))
        resized_image = cv2.resize(cropped_2, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)

        return resized_image

    else:
        rate = (temp_w - temp_h) / temp_w
        pis = int(256 * (1 - rate))
        predict_image = Image.fromarray(np.uint8(predict_image))
        cropped_1 = predict_image.crop((0, 0, 256, pis))  # (left, upper, right, lower)
        cropped_2 = np.array(cropped_1)
        resized_image = cv2.resize(cropped_2, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)

        return resized_image



