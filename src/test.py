from src.utils import to_cuda
from src.metrics import acc_metric, dice_metric, dice
from src.visualize import show_segmentation, predb_to_mask

import torch

from params import *
# todo: to finish
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

    # Should we transform the float predictions into class labels??
    predb_test = torch.empty(predb.shape[0], predb.shape[2], predb.shape[2])
    for i in range(predb.shape[0]):
        predb_test[i,:,:]=predb_to_mask(predb,i)

    # Evaluation - Dice score
    average_dice, class_dice = dice_metric(predb, yb)

    # Alessandro dice
    average_dice_A, class_dice_A = dice(predb_test, yb)
    baseline_dice =  0.607425 # [0.9652, 0.5956, 0.3764, 0.4925]
    print(f"Final average DICE score: {round(average_dice, 4)} {class_dice} (delta to baseline {round(average_dice - baseline_dice, 4)})")
    print(
        f"Final average DICE score (Ale: {round(average_dice_A, 4)} {class_dice_A} (delta to baseline {round(average_dice_A - baseline_dice, 4)})")
    
    return accuracy, average_dice, class_dice