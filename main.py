from src.data import load_train_test_val
from src.visualize import plot_loss_acc
from src.model import Unet2D
from src.train import train
from src.utils import save_result
from src.test import test
from src.metrics import acc_metric

from params import *

import torch
from torch import nn

import argparse
from datetime import datetime

def main(msg):
    train_data, test_data, valid_data = load_train_test_val(DATA_PARAMS, PREP_STEPS, TRAIN_TRANSFORMS)    
    
    # MODEL: Unet2D (one input channel, 4 output channels)
    # Outputs: Probabilities for each class for each pixel in different layer)
    unet = Unet2D(1, out_channels=4)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=LEARN_RATE)

    if LOAD:
        unet.load_state_dict(torch.load(MODEL_PATH+FILE_NAME))

    start = datetime.now()
    # Training process
    train_loss, valid_loss, train_acc, valid_acc = train(unet, train_data, valid_data, loss_fn, opt, acc_metric, epochs=EPOCHS)

    end = datetime.now()

    print("Elapsed time is {}".format(str(end-start)))

    plot_loss_acc(train_loss, valid_loss, train_acc, valid_acc)

    accuracy, average_dice, class_dice = test(unet, test_data, True)

    # save the result
    msg = 'Test the saving function'

    file_name = 'test1'

    # Save the model
    save_result(unet, file_name, accuracy, average_dice, class_dice, msg = msg )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument("-d", "--description", help="Description of the model", action=None)

    args = parser.parse_args()
    msg = args.description
    
    main(msg)