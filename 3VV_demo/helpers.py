# helpers.py
import os
import csv
import numpy as np


def mask_to_onehot(y_path, mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    # for colour in palette:
    if y_path.split('/')[-1].split('_')[0] == 'P':
        equality = np.equal(mask, palette[0])
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)

    elif y_path.split('/')[-1].split('_')[0] == 'A':
        equality = np.equal(mask, palette[1])
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)

    elif y_path.split('/')[-1].split('_')[0] == 'V':
        equality = np.equal(mask, palette[2])
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)

    else:
        equality_85 = np.equal(mask, 85)
        equality_170 = np.equal(mask, 170)
        equality_255 = np.equal(mask, 255)
        equality = equality_85 + equality_170 + equality_255
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)

    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def mask_to_onehot_three(mask_path, mask):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in [0, 85, 170, 255]:

        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)

    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)

    return semantic_map



def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x
