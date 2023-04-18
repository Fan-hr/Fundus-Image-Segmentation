import torch
import config
from Data2Set import MyDateSet
from tool import data_transforms, load_checkpoint
from Train_net import train_net
from torch.utils.data import DataLoader, random_split
from Net import UNet

# 生成数据集
Set = MyDateSet(transform=data_transforms, aug=True)
print(int(len(Set) * 0.8), int(len(Set) * 0.2))
Train_set, Value_set = random_split(dataset=Set, lengths=[int(len(Set) * 0.8), int(len(Set) * 0.2)],
                                    generator=torch.Generator().manual_seed(0))
train_data = DataLoader(Train_set, batch_size=config.BatchSize, shuffle=True)
value_data = DataLoader(Value_set, batch_size=config.BatchSize, shuffle=True)
# 初始化网络结构
net = UNet(n_classes=1, n_channels=3).to(device=config.device)
# 设置优化器
optimizer = torch.optim.Adam(list(net.parameters()), config.lr, betas=config.betas)
# 加载checkpoint权重
if config.check:
    load_checkpoint(model=net, optimizer=optimizer)
# 训练
train_net(net, train_data, value_data, optimizer)
