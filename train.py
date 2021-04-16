import numpy as np
import pandas as pd
import matplotlib as mp
from utils import *
import matplotlib.pyplot as plt
import time
import os
import argparse
from os.path import isfile
from datetime import datetime

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn
import torch.nn.functional as F

from DatasetLoader import DatasetLoader
from Unet2D import Unet2D
from PIL import Image

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, dice_metric, epochs=1):
    start = time.time()
    to_cuda(model)

    train_loss, valid_loss = [], []

    best_acc = 0.0
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs} [START]')
        print('-' * 40)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0
            running_dice = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                start_batch = time.time()
                y = to_cuda(y)
                x = to_cuda(x)
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())
                        
                # Compute stats (accuracy, dice score)
                acc = acc_fn(outputs, y)
                #average_dice, dice_scores = dice_metric(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 
                #running_dice += average_dice*dataloader.batch_size

                if step % 50 == 0:
                    print(f"Current step: {step}  Loss: {round(loss.item(), 4)}  Acc: {round(acc.item(), 4)}  "
                          f"[AllocMem (Mb): {round(torch.cuda.memory_allocated()/1024/1024, 4)}]")
                    # clear_output(wait=True)
                    # print(torch.cuda.memory_summary())
                    
                stop_batch = time.time() - start_batch
                #print('\nBatch complete in {:.0f}m {:.0f}s'.format(stop_batch // 60, stop_batch % 60))

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)
            #epoch_dice = running_dice / len(dataloader.dataset)

            print(f'--> Epoch {epoch+1}/{epochs} [PERFORMANCE: {phase}]')
            print('   ','-' * 20)
            print(f"    {phase} Loss: {round(epoch_loss.item(), 4)}  Acc: {round(epoch_acc.item(), 4)}  ")
                  #f"Dice score: {round(epoch_dice, 4)} --> {dice_scores}")
            print('   ','-' * 20)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return train_loss, valid_loss    

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
    
def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    #p = F.softmax(predb[idx], 0)
    #return p.argmax(0).cpu()
    return predb[idx].argmax(0).cpu()

def pre_processing_verbose(pre_processing_steps):
    if len(pre_processing_steps) > 1:
        print(f'The dataset will be undergoes the following pre-processing steps: ')
        for step in pre_processing_steps:
            print(f'{step}')
    return


def main(model_path, pretrained):
    # Batch size
    bs = 8 

    #epochs
    epochs_val =  50

    #learning rate
    learn_rate = 0.01

    #sets the matplotlib display backend (most likely not needed)
    #mp.use('TkAgg', force=True)
    
    #enable if you want to see some plotting
    visual_debug = False
    
    # Resolution of the image - Watch out Memory usage (Naive)
    # (1, 1.5 or 2, 3, 4) 
    scale = 1 
    image_resolution =  int(384 * scale)  

    # In this list define the sequentiality of the pre-processing steps
    # Recommended steps 'GaussBlur' and 'BilateralSmooth' (or both the image will be very smoothed)
    pre_processing_steps = []
    pre_processing_verbose(pre_processing_steps)

    # Load the data (raw and gt images)
    base_path = Path('Data') # /work/datasets/medical_project
    dataset = "extracted_CAMUS" # CAMUS_resized
    data = DatasetLoader(Path.joinpath(base_path, dataset, 'train_gray'), 
                         Path.joinpath(base_path, dataset, 'train_gt'), pre_processing_steps=pre_processing_steps,
                         image_resolution=image_resolution)
    
    # Split the training, test and validation datasets and initialize the data loaders
    train_size = int(0.8 * len(data))
    test_size = int(0.1 * len(data))
    val_size = int(0.1 * len(data))
    
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(data, (train_size, test_size, val_size))
    train_data = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=bs, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=bs, shuffle=True)
    print(f"\nImages loaded: {len(data)} [training: {len(train_dataset)}, test: {len(test_dataset)}, valid: {len(valid_dataset)}]")
    
    # Visualize shape of raw and ground true images
    xb, yb = next(iter(train_data))
    print(f"RAW IMAGES: {xb.shape}\n GT IMAGES: {yb.shape}")
    
    if visual_debug:
        idk_image = 150
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data.open_as_array(idk_image))
        ax[1].imshow(data.open_mask(idk_image))
        plt.show()

    # MODEL: U-Net 2D: Convolutional Networks for Biomedical Image Segmentation
    unet = Unet2D(in_channels=1, out_channels=4)
    #print(unet)
    
    if pretrained:
        unet.load_state_dict(torch.load(model_path + file_name))
    else:
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(unet.parameters(), lr=learn_rate)

        # Training process
        train_loss, test_loss = train(unet, train_data, test_data, loss_fn, opt, acc_metric, dice_metric, epochs=epochs_val)

        # Plot training and test losses
        if visual_debug:
            plt.figure(figsize=(10,8))
            plt.plot(train_loss, label='Train loss')
            plt.plot(test_loss, label='Test loss')
            plt.legend()
            plt.show()

    # Predict on the validation data
    xb, yb = next(iter(valid_data))
    with torch.no_grad():
        predb = unet(to_cuda(xb))
        
    # Evaluation - Accuracy
    accuracy = acc_metric(predb, yb).item()
    baseline_accuracy = 0.9705810547 # TRAINING TIME: 102m 6s
    print(f"\nFinal Accuracy: {round(accuracy, 4)} " 
          f"(delta to baseline {round(accuracy - baseline_accuracy, 4)})")
    
    # Evaluation - Dice score
    average_dice, class_dice = dice_metric(predb, yb)
    baseline_dice =  0.607425 # [0.9652, 0.5956, 0.3764, 0.4925]
    print(f"Final average DICE score: {round(average_dice, 4)} {class_dice} " 
          f"(delta to baseline {round(average_dice - baseline_dice, 4)})\n")
    
    # show the predicted segmentations
    if visual_debug:
        fig, ax = plt.subplots(bs,3, figsize=(15,bs*5))
        for i in range(bs):
            ax[i,0].imshow(batch_to_img(xb,i))
            ax[i,1].imshow(yb[i])
            ax[i,2].imshow(predb_to_mask(predb, i))
        plt.show()
    
    # Save the model
    if model_path is not None:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        # Save performance
        now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(model_path + "/performance.txt", "a") as text_file:
            print(f"data:{now}\ndataset:{dataset}\n"
                  f"epoch:{epochs_val}\nimage_resolution:{image_resolution}\n" 
                  f"pre_proc:{pre_processing_steps}\nacc:{round(accuracy, 4)}\n"
                  f"avg_dice:{round(average_dice, 4)}\nclass_dice_scores:{str(class_dice)}\n", 
                  file = text_file)
        
        # Save model
        torch.save(unet.state_dict(), model_path + file_name)
        print(f"Model state has been saved in /{model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train function')

    parser.add_argument("--model_path", help="Path of the model", default="Saved_model")
    parser.add_argument("--pretrained", help="If us an existing model", action="store_true")

    args = parser.parse_args()
    model_path = args.model_path
    file_name = "/unet_model.pt"
    pretrained = args.pretrained
    
    if pretrained:
        if model_path is None:
            raise FileNotFoundError('You have not given any model path')
        elif not isfile(model_path):
            raise FileNotFoundError('The model path is not valid, or not exists.')
    else:
        if model_path is None:
            print('Model is not saved')
        
    main(model_path, pretrained)
