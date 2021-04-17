import torch
import numpy as np

from datetime import datetime

from params import *

def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements

def pre_processing_verbose(pre_processing_steps):
    if len(pre_processing_steps) > 1:
        print(f'The dataset will be undergoes the following pre-processing steps: ')
        for step in pre_processing_steps:
            print(f'{step}')
    return

def save_result(model, model_name, accuracy, average_dice, class_dice, msg=None):
    # Save performance
    now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(MODEL_PATH + "performance.txt", "a") as text_file:
        print(f"data:{now}\ndataset:{DATA_PARAMS['dataset']}\n"
              f"epoch:{EPOCHS}\nimage_resolution:{DATA_PARAMS['image_resolution']}\n" 
              f"pre_proc:{PREP_STEPS}\nacc:{round(accuracy, 4)}\n"
              f"avg_dice:{round(average_dice, 4)}\nclass_dice_scores:{str(class_dice)}\n", 
              f"description:{msg}",
              file = text_file)

    # Save model
    torch.save(model.state_dict(), MODEL_PATH + model_name+'.pt')
    print(f"Model state has been saved in {MODEL_PATH+model_name}")