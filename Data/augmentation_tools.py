import cv2
import random
import numpy as np
import math


def flip(img, label):
    choose = random.uniform(0., 1.)
    if 0 <= choose < 0.32:
        way = 1
    elif 0.32 <= choose < 0.65:
        way = 0
    else:
        way = -1
    flip_img = cv2.flip(src=img, flipCode=way)
    flip_label = cv2.flip(src=label, flipCode=way)
    return flip_img, flip_label


def shift(img, label):
    # 随机均匀产生偏移量
    x = random.uniform(-(img.shape[1] - 1) / 2, (img.shape[1] - 1) / 2)
    y = random.uniform(-(img.shape[0] - 1) / 2, (img.shape[0] - 1) / 2)
    # 构造放射变换矩阵
    A = np.float32([[1, 0, x], [0, 1, y]])
    # 平移图像
    shift_img = cv2.warpAffine(img, A, (img.shape[0], img.shape[1]))
    shift_label = cv2.warpAffine(label, A, (img.shape[0], img.shape[1]))
    return shift_img, shift_label


def rotate(img, label, angle=5, scale=1.0):
    h = img.shape[0]
    w = img.shape[1]
    Wrangle = np.deg2rad(angle)
    new_h = (abs(np.cos(Wrangle) * h) + abs(np.sin(Wrangle) * w)) * scale
    new_w = (abs(np.sin(Wrangle) * h) + abs(np.cos(Wrangle) * w)) * scale
    rot_mat = cv2.getRotationMatrix2D((new_w * 0.5, new_h * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(new_w - w) * 0.5, (new_h - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(new_w)), int(math.ceil(new_h))), flags=cv2.INTER_LANCZOS4)
    rot_label = cv2.warpAffine(label, rot_mat, (int(math.ceil(new_w)), int(math.ceil(new_h))), flags=cv2.INTER_LANCZOS4)
    return rot_img, rot_label


def crop(img, label):
    # 随机均匀生成裁剪量
    crop_x_min = int(random.uniform(img.shape[1] / 8, img.shape[1] / 2))
    crop_y_min = int(random.uniform(img.shape[0] / 8, img.shape[0] / 2))
    crop_x_max = int(random.uniform(img.shape[1] / 2, img.shape[1]))
    crop_y_max = int(random.uniform(img.shape[0] / 2, img.shape[0]))
    # crop
    crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    crop_label = label[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    return crop_img, crop_label
