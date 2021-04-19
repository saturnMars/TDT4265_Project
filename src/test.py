from src.utils import to_cuda
from src.metrics import acc_metric, dice_metric, dice, classwise_f1
from src.visualize import show_segmentation, predb_to_mask

import torch

from params import *

# todo: to finish

def test(model, data, visualize=False):

    # Predict on the (validation) data
    xb, yb = next(iter(data))
    with torch.no_grad():
        predb = model(to_cuda(xb))

    if visualize:
        show_segmentation(xb, yb, predb)
        
    # Evaluation - Accuracy
    accuracy = acc_metric(predb, yb).item()
    baseline_accuracy = 0.95 
    print(f"\nFinal Accuracy: {round(accuracy, 4)} (delta to baseline {round(accuracy - baseline_accuracy, 4)})")

    # Evaluation - Dice coefficient
    baseline_dice =  100
    
    # IMPLEMENTATION 1: OLD  
    average_dice, class_dice = dice_metric(predb, yb)

    # IMPLEMENTATION 2: Alessandro
    predb_test = torch.empty(predb.shape[0], predb.shape[2], predb.shape[2])
    for i in range(predb.shape[0]):
        predb_test[i, :, :]=predb_to_mask(predb, i)
    average_dice_A, class_dice_A = dice(predb_test, yb)

    #IMPLEMENTATION 3: Github repo (Dice = f1)
    average_dice_f, class_dice_f =  classwise_f1(predb, yb)
   
    print(f"Final average DICE score (OLD): {round(average_dice, 4)} {class_dice} (delta to baseline {round(average_dice - baseline_dice, 4)})")
    print(f"Final average DICE score (Ale): {round(average_dice_A, 4)} {class_dice_A} (delta to baseline {round(average_dice_A - baseline_dice, 4)})")
    print(f"Final average DICE score (GitHub - F1): {round(average_dice_f, 4)} {class_dice_f} (delta to baseline {round(average_dice_f - baseline_dice, 4)})")
  
    return accuracy, average_dice_f, class_dice_f