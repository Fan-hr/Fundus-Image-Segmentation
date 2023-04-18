import torch
import config
from torchvision.transforms import transforms

data_transforms = transforms.Compose([transforms.ToTensor()])


def save_checkpoint(model, optimizer, Epoch):
    print("=> saving checkpoint")
    point = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(point, config.model_path + config.model_info["name"] + "_epoch" + str(Epoch) + ".pth")


def load_checkpoint(model, optimizer):
    print("=> loading checkpoint")
    point = torch.load(
        config.model_path + config.model_info["name"] + "_epoch" + str(config.model_info["epoch"]) + ".pth",
        map_location=config.device
    )
    model.load_state_dict(point["state_dict"])
    optimizer.load_state_dict(point["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = config.lr


def dice(inputs, targets, smooth=1e-4):
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = 2 * intersection / (inputs.sum() + targets.sum() + smooth)
    return dice
