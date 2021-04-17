import matplotlib.pyplot as plt

def plot_loss_acc(train_loss, test_loss, train_acc, test_acc):
    # Plot training and test losses
    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='Test loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,8))
    plt.plot(train_acc, label='Train accuracy')
    plt.plot(test_acc, label='Test accuracy')
    plt.legend()
    plt.show()

def other_visualization():
    raise NotImplemented


    # show the predicted segmentations
    if visual_debug:
        fig, ax = plt.subplots(bs,3, figsize=(15,bs*5))
        for i in range(bs):
            ax[i,0].imshow(batch_to_img(xb,i))
            ax[i,1].imshow(yb[i])
            ax[i,2].imshow(predb_to_mask(predb, i))
        plt.show()

    if visual_debug:
        idk_image = 150
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(data.open_as_array(idk_image))
        ax[1].imshow(data.open_mask(idk_image))
        plt.show()
