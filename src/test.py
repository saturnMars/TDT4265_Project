from src.utils import to_cuda
from src.metrics import acc_metric, dice_metric
from src.visualize import show_segmentation, predb_to_mask

import torch

from params import *

# todo: to finish

def test(model, data, visualize=False):
    # Predict on the validation data
    
    if visualize:
        xb, yb = next(iter(data))
        with torch.no_grad():
            predb = model(to_cuda(xb))
    
        show_segmentation(xb, yb, predb)
    
    accuracy = 0
    average_dice = 0
    class_dice = torch.zeros([4])
    for xb, yb in data:
        with torch.no_grad():
            predb = model(to_cuda(xb))
        
        accuracy += acc_metric(predb, yb).item()
        temp_av_dice, temp_class_dice = dice_metric(predb, yb)
        average_dice += temp_av_dice
        class_dice += temp_class_dice
    
    accuracy /= len(data)
    average_dice /= len(data)
    class_dice /= len(data)
    
    average_dice = average_dice.numpy()[()]
    class_dice = class_dice.numpy().tolist()
    
    # Evaluation - Accuracy
    baseline_accuracy = 0.9705810547 # TRAINING TIME: 102m 6s
    print(f"\nFinal Accuracy: {round(accuracy, 4)} (delta to baseline {(accuracy - baseline_accuracy, 4)})")

    # Evaluation - Dice score
    baseline_dice =  0.607425 # [0.9652, 0.5956, 0.3764, 0.4925]
    print(f"Final average DICE score: {round(average_dice, 4)} {class_dice} (delta to baseline {round(average_dice - baseline_dice, 4)})")
    
    return accuracy, average_dice, class_dice