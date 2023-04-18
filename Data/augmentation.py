import tqdm
import cv2
import os
from augmentation_tools import crop, shift, rotate, flip

image_path = os.listdir("./Seg_train/images/")
label_path = os.listdir("./Seg_train/1st_manual/")
loop = tqdm.tqdm(image_path, leave=True)
for i, item in enumerate(loop):
    img = cv2.imread("./Seg_train/images/" + item)
    print(img.shape)
    label = cv2.imread("./Seg_train/1st_manual/" + label_path[i])
    # 保存原始图像
    cv2.imwrite("./Seg_train_aug/images/" + str(i) + '_ori_' + item, img)
    cv2.imwrite("./Seg_train_aug/1st_manual/" + str(i) + '_ori_' + label_path[i][:-5]+'.png', label)
    # 随机沿坐标轴翻转
    flip_img, flip_label = flip(img, label)
    cv2.imwrite("./Seg_train_aug/images/" + str(i) + '_flip_' + item, flip_img)
    cv2.imwrite("./Seg_train_aug/1st_manual/" + str(i) + '_flip_' + label_path[i][:-5]+'.png', flip_label)
    # 随机平移
    shift_img, shift_label = shift(img, label)
    cv2.imwrite("./Seg_train_aug/images/" + str(i) + '_shift_' + item, shift_img)
    cv2.imwrite("./Seg_train_aug/1st_manual/" + str(i) + '_shift_' + label_path[i][:-5]+'.png', shift_label)
    # 随机旋转
    rotate_img, rotate_label = rotate(img, label)
    cv2.imwrite("./Seg_train_aug/images/" + str(i) + '_rotate_' + item, rotate_img)
    cv2.imwrite("./Seg_train_aug/1st_manual/" + str(i) + '_rotate_' + label_path[i][:-5]+'.png', rotate_label)
    # 随机裁剪
    crop_img, crop_label = crop(img, label)
    cv2.imwrite("./Seg_train_aug/images/" + str(i) + '_crop_' + item, crop_img)
    cv2.imwrite("./Seg_train_aug/1st_manual/" + str(i) + '_crop_' + label_path[i][:-5]+'.png', crop_label)
