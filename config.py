import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_path = "./Data/Seg_train/1st_manual/"
image_path = "./Data/Seg_train/images/"
aug_label_path = "./Data/Seg_train_aug/1st_manual/"
aug_image_path = "./Data/Seg_train_aug/images/"
model_path = "./Model/"
check = False
lr = 0.0005
BatchSize = 1
Epoch = 50
betas = (0.9, 0.999)
model_info = {"name": "u-net", "epoch": 0}
