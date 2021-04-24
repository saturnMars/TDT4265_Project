import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time
import argparse
from os.path import isfile

import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn
import torch.nn.functional as F

from PIL import Image

from src.metrics import acc_metric, dice_metric
from src.utils import *

def validation(model, valid_dl, loss_fn, acc_fn):
    running_loss = 0.0
    running_acc = 0.0
    
    model.train(False)

    for x, y in valid_dl:
        y = to_cuda(y)
        x = to_cuda(x)

        with torch.no_grad():
            outputs = model(x)
            loss = loss_fn(outputs, y.long())
            
        acc = acc_fn(outputs, y)
        dice_acc = dice_metric(outputs, y)
            
        running_acc  += acc*valid_dl.batch_size
        running_loss += loss*valid_dl.batch_size
    model.train(True)

    return running_loss / len(valid_dl.dataset), running_acc / len(valid_dl.dataset), dice_acc

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    to_cuda(model)

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    best_acc = 0.0

    model.train(True)  # Set trainind mode = true
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        

        running_loss = 0.0
        running_acc = 0.0

        step = 0

        # iterate over data
        for x, y in train_dl:
            y = to_cuda(y)
            x = to_cuda(x)
            step += 1

            # forward pass
            # zero the gradients
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)

            # the backward pass frees the graph memory, so there is no 
            # need for torch.no_grad in this training pass
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # stats - whatever is the phase
            acc = acc_fn(outputs, y)
            dice = dice_metric(outputs, y)
            
            with torch.no_grad():
                running_acc  += acc*train_dl.batch_size
                running_loss += loss*train_dl.batch_size 
                if step % 40 == 0:
                    # clear_output(wait=True)
                    v_loss, v_acc, v_dice = validation(model, valid_dl, loss_fn, acc_fn)

                    valid_loss.append(v_loss)
                    valid_acc.append(v_acc)

                    train_loss.append(loss)
                    train_acc.append(acc)
                    print(f'Current step: {step}  Loss: {loss:.4f}  Acc: {acc:.3f} Dice: {dice} Val Loss: {v_loss:.4f} Val Acc: {v_acc:.3f} Val Dice: {v_dice}')

        epoch_loss = running_loss / len(train_dl.dataset)
        epoch_acc = running_acc / len(train_dl.dataset)

        v_loss, v_acc, v_dice = validation(model, valid_dl, loss_fn, acc_fn)

        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)
        print('Train Loss: {:.4f} Acc: {}'.format(epoch_loss, epoch_acc))
        print('-' * 10)
        
        valid_loss.append(v_loss)
        valid_acc.append(v_acc)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
    return train_loss, valid_loss, train_acc,  valid_acc 
