import torch
import config
import torch.nn.functional as F
from tool import dice, save_checkpoint
from Net import DiceBCELoss


def train_net(net, train_data, value_data, optimizer):
    loss_function = DiceBCELoss().to(device=config.device)
    # ne_opt = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)
    Scaler = torch.cuda.amp.GradScaler()
    for e in range(1, config.Epoch + 1):
        dice_score = 0
        train_loss = 0
        value_loss = 0
        net.train()
        for img, mask_true in train_data:
            img = torch.tensor(img.to(config.device), dtype=torch.float32)
            mask_true = torch.abs(torch.tensor(mask_true.to(config.device), dtype=torch.float))
            # forward
            with torch.cuda.amp.autocast():
                mask_pre = net(img)
                loss = loss_function(mask_pre, mask_true)
                # backward
            optimizer.zero_grad()
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            train_loss += loss.data
        net.eval()
        for img, mask_true in value_data:
            img = torch.tensor(img.to(config.device), dtype=torch.float32)
            mask_true = torch.abs(torch.tensor(mask_true.to(config.device), dtype=torch.float))
            mask_pre = net(img)
            loss = loss_function(mask_pre.float(), mask_true.float())
            value_loss += loss.data
            mask_pre = (F.sigmoid(mask_pre) > 0.5).float()
            print(mask_pre.sum())
            dice_score += dice(mask_pre, mask_true)
        print('-----epoch:{},lr:{}----------'.format(e, optimizer.state_dict()['param_groups'][0]['lr']))
        print('Train Loss:{},Value Loss:{},dice:{}'.format(train_loss / len(train_data),
                                                           value_loss / len(value_data),
                                                           dice_score / len(value_data)))
        if e % 10 == 0:
            save_checkpoint(net, optimizer, e)
