import torch.nn.functional as F
from src.utils import *


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == to_cuda(yb)).float().mean()

def dice_metric(predb, yb, smooth = 1e-5):
    batch_size = predb.shape[0]
    num_classes = predb.shape[1]
    
    dice_scores = []
    for class_idk in range(num_classes):
        # Flatten
        pred = predb[:, class_idk, :, :].view(batch_size, -1)
        target = to_cuda(yb).view(batch_size, -1)

        intersection = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1)
        dice = 1 - ((2 * intersection + smooth) / (union + smooth))

        #tp = torch.sum(target * pred, dim = 1)
        #fp = torch.sum(pred, dim = 1) - tp
        #fn = torch.sum(target, dim = 1) - tp
        #dice = 1 - ((2*tp + smooth) / (2*tp + fp + fn + smooth))
            
        # Compute the average value for the batch
        class_dice =  torch.sum(dice)/batch_size
        dice_scores.append(round(class_dice.item(), 4))
        
    # Compute the average score among all classes
    average_dice = sum(dice_scores)/num_classes
    return average_dice, dice_scores



def dice(predb, yb, n_classes=4, smooth=1e5):
  batch_size = predb.shape[0]
  dice = np.zeros((n_classes, batch_size))

  for batch in range(batch_size):
      pred = predb[batch, :, :].view(-1)
      target = yb[batch, :, :].view(-1)

      # Ignore IoU for background class ("0")
      for cls in range(n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        current_cls_pred = (pred == cls).float()
        current_cls_target = (target == cls).float()
        intersection = (current_cls_pred * current_cls_target).sum()
        #intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = current_cls_pred.sum() + current_cls_target.sum() - intersection
        dice[cls, batch] = dice[cls,batch] + ((2 * intersection + smooth) / (union + smooth)).item()

  dice_scores = np.mean(dice, axis=1)/n_classes
  return np.mean(dice_scores), list(dice_scores)
