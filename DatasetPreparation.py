import numpy as np
import pandas as pd
import matplotlib as mp
from utils import *
import matplotlib.pyplot as plt
import time
from PIL import Image
from medimage import image
import os
from pathlib import Path

# todo: We can add function to extract also the TEE images from the hdf5 format
def prepare_image_data_gt(directory_train, patient, type):
    """
    This function is used to extract ES ED pair of images data and it's ground truth image's labels, from the CAMUS
    dataset
    :param directory_train: string
        Path where you find there are all the patient folders
    :param patient: string
        Current patient folder
    :param type: string
        Current image's type that you want to extract ['2CH_ED', '2CH_ES','4CH_ED','4CH_ES']
    :return: image_data, image_gt
    """
    path_data = os.path.join(directory_train, patient, patient + '_' + type +'.mhd')
    path_gt = os.path.join(directory_train, patient, patient + '_' + type + '_gt.mhd')

    # Retrieve the data
    image_data = image(path_data)
    image_gt = image(path_gt)
    return image_data, image_gt


def save_medimage_to_PIL(image, main_directory, patient, image_type, type='gray'):
    """
    Function used to convert from medimage formato to a tif format used for the training
    :param image: medimage format
    :param main_directory: Directory where to save the image
    :param patient: patient string
    :param image_type: choose the image_type (['2CH_ED', '2CH_ES','4CH_ED','4CH_ES'])
    :param type: string can be 'gray' or 'gt'
    :return: Nothing
    """
    # Converting image into PIL image
    PIL_image = Image.fromarray(np.uint8(image.imdata.squeeze())).convert('RGB')

    if type == 'gray':
        type_path = 'train_gray'

    elif type == 'gt':
        type_path = 'train_gt'

    else:
        raise Exception('Unknown data type')

    saving_path = os.path.join(main_directory, type_path, type + '_' + patient + '_' + image_type + '.tif')
    PIL_image.save(saving_path)
    return


def main():

    # Directory where there is the TRAINING CAMUS DATASET
    directory = Path('training')
    # Directory where i would like to store the images extracted from the CAMUS
    saving_directory = Path('extracted_CAMUS')

    # Loop through all the patients
    for patient in os.listdir(directory):

        # Paths regarding the data of the US images -> 2CH_ED
        image_data_2CH_ED, image_gt_2CH_ED = prepare_image_data_gt(directory, patient, '2CH_ED')
        # Paths regarding the data of the US images -> 2CH_ES
        image_data_2CH_ES, image_gt_2CH_ES = prepare_image_data_gt(directory, patient, '2CH_ES')
        # Paths regarding the data of the US images -> 4CH_ED
        image_data_4CH_ED, image_gt_4CH_ED = prepare_image_data_gt(directory, patient, '4CH_ED')
        # Paths regarding the data of the US images -> 4CH_ES
        image_data_4CH_ES, image_gt_4CH_ES = prepare_image_data_gt(directory, patient, '4CH_ES')

        # Saving Images (GRAY DATA)
        save_medimage_to_PIL(image_data_2CH_ED, saving_directory, patient, '2CH_ED', type='gray')
        save_medimage_to_PIL(image_data_2CH_ES, saving_directory, patient, '2CH_ES', type='gray')
        save_medimage_to_PIL(image_data_4CH_ED, saving_directory, patient, '4CH_ED', type='gray')
        save_medimage_to_PIL(image_data_4CH_ES, saving_directory, patient, '4CH_ES', type='gray')

        # Saving Images (GROUND TRUTH DATA)
        save_medimage_to_PIL(image_gt_2CH_ED, saving_directory, patient, '2CH_ED', type='gt')
        save_medimage_to_PIL(image_gt_2CH_ES, saving_directory, patient, '2CH_ES', type='gt')
        save_medimage_to_PIL(image_gt_4CH_ED, saving_directory, patient, '4CH_ED', type='gt')
        save_medimage_to_PIL(image_gt_4CH_ES, saving_directory, patient, '4CH_ES', type='gt')

if __name__ == "__main__":
    main()