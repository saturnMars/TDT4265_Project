import numpy as np
import pandas as pd
import matplotlib as mp
from utils import *
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageOps
from medimage import image
import os
import h5py
import warnings
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
    Function used to convert from medimage formato to a tif format used for the training, before there will be a padding
    in order to have a square image
    :param image: medimage format
    :param main_directory: Directory where to save the image
    :param patient: patient string
    :param image_type: choose the image_type (['2CH_ED', '2CH_ES','4CH_ED','4CH_ES'])
    :param type: string can be 'gray' or 'gt'
    :return: Nothing
    """
    # Converting image into PIL image
    PIL_image = Image.fromarray(np.uint8(image.imdata.squeeze())).convert('RGB')
    # Max dimension
    max_dimension = min(PIL_image.size)

    # Resize with padding
    #delta_w = max_dimension - PIL_image.size[1]
    #delta_h = max_dimension - PIL_image.size[0]
    #padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    #new_im = ImageOps.expand(PIL_image, padding)
    #PIL_image = PIL_image.resize((max_dimension, max_dimension))

    if type == 'gray':
        type_path = 'train_gray'

    elif type == 'gt':
        type_path = 'train_gt'

    else:
        raise Exception('Unknown data type')

    saving_path = os.path.join(main_directory, type_path, type + '_' + patient + '_' + image_type + '.tif')
    PIL_image.save(saving_path)
    return


def equal_pixel(image_data, image_gt, pixel_spacing=None):
    """
    Function used to resample the image in order to have a isotropic image
    :param image_data: Medimage data that have a gray data
    :param image_gt: Ground truth image
    :param pixel_spacing: (float) Define a costum pixel_spacing in the image in order to have iso-tropic image
    :return: prepro_image_data: Resampled image with the new pixel_spacing
    :return: prepro_image_gt: Resampled ground truth image with the new pixel_spacing
    """
    # Pixel spacing
    element_spacing = image_data.spacing()

    if not pixel_spacing:
        pixel_spacing = min(element_spacing)

    # Resample the image in order to have the same pixel size
    prepro_image_data, prepro_image_gt = image_data.copy(), image_gt.copy()

    prepro_image_data.resample([pixel_spacing, pixel_spacing, element_spacing[2]])
    prepro_image_gt.resample([pixel_spacing, pixel_spacing, element_spacing[2]])
    return prepro_image_data, prepro_image_gt


def main():

    # Directory where there is the TRAINING CAMUS DATASET / TEE
    directory_tte = Path('training')
    directory_tee = Path('data/TEE')
    # Directory where i would like to store the images extracted from the CAMUS
    saving_directory = Path('extracted_CAMUS')

    flag_equal_pixel = True
    # Remove all the Warning after developing
    warnings.filterwarnings("ignore")

    # Loop through all the patients (EXTRACTION OF THE TEE)
    for patient in os.listdir(directory_tee):

        # Looping through all the hdf5 file for each patient
        for current_hdf5 in os.listdir(os.path.join(directory_tee, patient)):
            path_hdf5 = os.path.join(directory_tee, patient, current_hdf5)
            image_hdf5 = h5py.File(path_hdf5, 'r')

    # You could generate 15 gb of images
    raise Exception('DO NOT RUN THE CODE cause resize function not implemented yet')

    # Loop through all the patients (EXTRACTION OF THE TTE)
    for patient in os.listdir(directory_tte):

        # Paths regarding the data of the US images -> 2CH_ED
        image_data_2CH_ED, image_gt_2CH_ED = prepare_image_data_gt(directory_tte, patient, '2CH_ED')
        # Paths regarding the data of the US images -> 2CH_ES
        image_data_2CH_ES, image_gt_2CH_ES = prepare_image_data_gt(directory_tte, patient, '2CH_ES')
        # Paths regarding the data of the US images -> 4CH_ED
        image_data_4CH_ED, image_gt_4CH_ED = prepare_image_data_gt(directory_tte, patient, '4CH_ED')
        # Paths regarding the data of the US images -> 4CH_ES
        image_data_4CH_ES, image_gt_4CH_ES = prepare_image_data_gt(directory_tte, patient, '4CH_ES')

        if flag_equal_pixel:
            image_data_2CH_ED, image_gt_2CH_ED = equal_pixel(image_data_2CH_ED, image_gt_2CH_ED, pixel_spacing=None)
            image_data_2CH_ES, image_gt_2CH_ES = equal_pixel(image_data_2CH_ES, image_gt_2CH_ES, pixel_spacing=None)
            image_data_4CH_ED, image_gt_4CH_ED = equal_pixel(image_data_4CH_ED, image_gt_4CH_ED, pixel_spacing=None)
            image_data_4CH_ES, image_gt_4CH_ES = equal_pixel(image_data_4CH_ES, image_gt_4CH_ES, pixel_spacing=None)

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