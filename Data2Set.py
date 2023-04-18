import os
import config
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MyDateSet(Dataset):
    def __init__(self, transform, aug):
        self.transform = transform
        if aug:
            self.label_list = os.listdir(config.aug_label_path)
            self.img_list = os.listdir(config.aug_image_path)
        else:
            self.label_list = os.listdir(config.label_path)
            self.img_list = os.listdir(config.image_path)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_label = self.label_list[index]
        img = Image.open(config.image_path + img_path)
        label = Image.open(config.label_path + img_label).convert("1")
        img = self.process(img=img, is_mask=False)
        label = self.process(img=label, is_mask=True)
        img = self.transform(img)
        label = self.transform(label)
        return img, label

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def process(img, is_mask):
        # img = img.resize((512, 512), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_array = np.asarray(img)

        if not is_mask:
            img_array = img_array / 255
        return img_array
