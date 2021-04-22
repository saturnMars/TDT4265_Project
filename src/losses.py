from src.metrics import dice
import torch.nn.functional as F

def dice_loss(logits, true):
    dice_loss_mean, dice_loss_class = dice(logits, true)

    return (1-dice_loss_mean)

def cross_entropy_loss(logits, true):
    return F.cross_entropy(logits, true, reduction='mean')

def dice_cross_entropy_loss(logits, true):
    return cross_entropy_loss(logits, true)+dice_loss(logits, true)


