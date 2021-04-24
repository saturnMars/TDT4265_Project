import matplotlib.pyplot as plt
import numpy as np

from params import *

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))[:, :, 0]

def predb_to_mask(predb, idx):
    #p = F.softmax(predb[idx], 0)
    #return p.argmax(0).cpu()
    return predb[idx].argmax(0).cpu()

def plot_loss_acc(train_loss, test_loss, train_acc, test_acc):
    # Plot training and test losses
    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='Valid loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,8))
    plt.plot(train_acc, label='Train accuracy')
    plt.plot(test_acc, label='Valid accuracy')
    plt.legend()
    plt.show()

def show_segmentation(x, y, pred):
    bs = DATA_PARAMS['batch_size']
    fig, ax = plt.subplots(bs, 3, figsize=(15,bs*5))
    for i in range(bs):
        ax[i,0].imshow(batch_to_img(x,i))
        ax[i,1].imshow(y[i])
        ax[i,2].imshow(predb_to_mask(pred, i))
    plt.show()

    
def other_visualization():
    raise NotImplementeda
    # show the predicted segmentations

    if visual_debug:
        idk_image = 150
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data.open_as_array(idk_image))
        ax[1].imshow(data.open_mask(idk_image))
        plt.show()
