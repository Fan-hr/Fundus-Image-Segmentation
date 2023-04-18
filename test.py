import torch
import config
import torch.nn.functional as F
import torchvision.utils as utils
from tool import data_transforms
from Data2Set import MyDateSet
from torch.utils.data import DataLoader
from Net import UNet

# 加载数据
Value_set = MyDateSet(transform=data_transforms, aug=False)
value_data = DataLoader(Value_set, batch_size=1)
# 加载模型
net = UNet(n_classes=1, n_channels=3).to(config.device)
point = torch.load(config.model_path + "u-net_epoch20.pth", map_location=config.device)
net.load_state_dict(point["state_dict"])
# 测试模型
net.eval()
for index, (img, _) in enumerate(value_data):
    img = torch.tensor(img.to(config.device), dtype=torch.float32)
    mask_pre = net(img)
    mask_pre = (F.sigmoid(mask_pre) > 0.5).float()
    utils.save_image(mask_pre, "./res/index-%d.png" % index)
torch.save(net, './u-net.pth')
