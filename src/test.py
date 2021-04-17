from src.utils import to_cuda
from src.metrics import acc_metric, dice_metric
from src.visualize import show_segmentation

import torch

from params import *

def test(model, data, visualize=False):
    # Predict on the validation data
    xb, yb = next(iter(data))
    with torch.no_grad():
        predb = model(to_cuda(xb))

    if visualize:
        show_segmentation(xb, yb, predb)
        
    # Evaluation - Accuracy
    accuracy = acc_metric(predb, yb).item()
    baseline_accuracy = 0.9705810547 # TRAINING TIME: 102m 6s
    print(f"\nFinal Accuracy: {round(accuracy, 4)} (delta to baseline {round(accuracy - baseline_accuracy, 4)})")

    # Evaluation - Dice score
    average_dice, class_dice = dice_metric(predb, yb)
    baseline_dice =  0.607425 # [0.9652, 0.5956, 0.3764, 0.4925]
    print(f"Final average DICE score: {round(average_dice, 4)} {class_dice} (delta to baseline {round(average_dice - baseline_dice, 4)})")
    
    return accuracy, average_dice, class_dice