import torch.nn.functional as F
from src.utils import *


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == to_cuda(yb)).float().mean()

def dice_metric(logits, true, eps=1e-7):
    num_classes = logits.shape[1]

    logits = torch.eye(num_classes)[logits.argmax(1)].permute(0, 3, 1, 2)
    
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()

    true_1_hot = true_1_hot.type(logits.type())

    dims = (2,3)
    
    # calculate TP
    intersection = torch.sum(logits * true_1_hot, dims)

    # calculate 2TP+FN+FP
    cardinality = torch.sum(logits + true_1_hot, dims)

    dice_loss_class = (2. * intersection / (cardinality + eps)).mean(0)
    dice_loss_mean = dice_loss_class.mean()
    
    return dice_loss_mean, dice_loss_class