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
from scipy.signal import find_peaks


# todo: We can add function to extract also the TEE images from the hdf5 format
# todo: Remain the extraction of the ground truth
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


def save_medimage_to_PIL(image, main_directory, patient, image_type, type='gray', resize_dim=384):
    """
    Function used to convert from medimage formato to a tif format used for the training, before there will be a padding
    in order to have a square image and a resize
    :param image: medimage format (TTE Image)
    :param main_directory: Directory where to save the image
    :param patient: patient string
    :param image_type: choose the image_type (['2CH_ED', '2CH_ES','4CH_ED','4CH_ES'])
    :param type: string can be 'gray' or 'gt'
    :param resize_dim: int Decide the dimension of the final image. (384 default for the model)
    :return: Nothing
    """
    # Converting image into PIL image
    PIL_image = Image.fromarray(np.uint8(image.imdata.squeeze())).convert('L')
    # Padding follows the bigger dimension
    max_dimension = max(PIL_image.size)

    # Padding
    #delta_w = max_dimension - PIL_image.size[0]
    #delta_h = max_dimension - PIL_image.size[1]
    #padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    #PIL_image = ImageOps.expand(PIL_image, padding)

    # This line one of code it will make isotropic
    PIL_image = PIL_image.resize((max_dimension, max_dimension))

    # Resize
    PIL_image = PIL_image.resize((resize_dim, resize_dim))

    if type == 'gray':
        type_path = 'train_gray'

    elif type == 'gt':
        type_path = 'train_gt'

    else:
        raise Exception('Unknown data type')

    saving_path = os.path.join(main_directory, type_path, type + '_' + patient + '_' + image_type + '.tif')
    PIL_image.save(saving_path, )
    return


def equal_pixel(image_data, image_gt, pixel_spacing=None):
    """
    Function used to resample the image in order to have a isotropic image
    :param image_data: Medimage data that have a gray data (TTE image)
    :param image_gt: Ground truth image
    :param pixel_spacing: (float) Define a costum pixel_spacing in the image in order to have iso-tropic image
    :return: prepro_image_data: Resampled image with the new pixel_spacing
    :return: prepro_image_gt: Resampled ground truth image with the new pixel_spacing
    """
    # Pixel spacing - this should be the distance between pixels
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
    resize_dim = 384
    # Hearbeat duration according to the literature
    heartbeat_duration = 0.8    # [sec]
    # Type of image for tte
    type_tte = ['2CH_ED', '2CH_ES', '4CH_ED', '4CH_ES']

    # Directory where i would like to store the images extracted from the CAMUS
    saving_directory_tte = Path('data\extracted_CAMUS')
    saving_directory_tee = Path('data\extracted_TEE')

    flag_equal_pixel = False # Leave it as FALSE cause wrong function
    # Remove all the Warning after developing
    warnings.filterwarnings("ignore")

    # ################## Loop through all the patients (EXTRACTION OF THE TEE) ##################
    for patient in os.listdir(directory_tee):

        # Looping through all the hdf5 file for each patient
        for current_hdf5 in os.listdir(os.path.join(directory_tee, patient)):
            path_hdf5 = os.path.join(directory_tee, patient, current_hdf5)
            image_hdf5 = h5py.File(path_hdf5, 'r')
            # Extract the tissue
            tissue_hdf5 = image_hdf5['tissue']
            tissue_data = tissue_hdf5['data']

            # Extract the ECG
            ecg_hdf5 = image_hdf5['ecg']
            ecg_data = ecg_hdf5['ecg_data']
            ecg_times = ecg_hdf5['ecg_times']

            index_list = [i for i in range(ecg_data.shape[0])]
            print('Lenght of the ecg signal: ' + str(len(index_list)))
            print('Number of image : ' + str(tissue_data.shape[2]))
            ecg_data_arr = ecg_data.__getitem__(index_list)
            ecg_times_arr = ecg_times.__getitem__(index_list)

            delta_t = ecg_times_arr[1]-ecg_times_arr[0] # Sampling frequency
            fs = np.round(1/delta_t)
            heartbeat_n_samples = np.round(fs*heartbeat_duration)

            # find the peaks - for the ED event
            max_point = np.max(ecg_data_arr)
            peaks, _ = find_peaks(ecg_data_arr, height=max_point/2, distance=heartbeat_n_samples/2)
            #min, _ = find_peaks(-ecg_data_arr, distance=heartbeat_n_samples / 2)
            # Plotting the ecg
            plt.plot(ecg_data_arr)
            plt.plot(peaks, ecg_data_arr[peaks], "x")
            #plt.plot(min, ecg_data_arr[min], "o")

            # Define manually the ES points
            if len(peaks)-1 > 0:
                min = plt.ginput(n=len(peaks)-1, show_clicks=True)
            plt.show()

            for peak_pos in peaks:
                idx = int(np.round(tissue_data.shape[2]*peak_pos/len(index_list)))

                # TEE image - ED event
                PIL_image_ED = Image.fromarray(np.uint8(tissue_data[:, :, idx].squeeze())).convert('L')

                # Padding follows the bigger dimension
                max_dimension = max(PIL_image_ED.size)

                # Make the image isotropic
                PIL_image_ED = PIL_image_ED.resize((max_dimension, max_dimension))

                # Resize
                PIL_image_ED = PIL_image_ED.resize((resize_dim, resize_dim))

                # Saving image (gray image)
                saving_path_ED = os.path.join(saving_directory_tee, 'train_gray', 'gray_' + current_hdf5[:-3] + '_' + str(idx) + 'ED.tif')
                PIL_image_ED.save(saving_path_ED)

            # Extract ES IMAGES
            for es_pos in min:
                idx = int(np.round(tissue_data.shape[2]*es_pos[0]/len(index_list)))

                # TEE image - ES event
                PIL_image_ES = Image.fromarray(np.uint8(tissue_data[:, :, idx].squeeze())).convert('L')

                # Padding follows the bigger dimension
                max_dimension = max(PIL_image_ES.size)

                # Make the image isotropic
                PIL_image_ES = PIL_image_ES.resize((max_dimension, max_dimension))

                # Resize
                PIL_image_ES = PIL_image_ES.resize((resize_dim, resize_dim))

                # Saving image (gray image)
                saving_path_ES = os.path.join(saving_directory_tee, 'train_gray', 'gray_' + current_hdf5[:-3] + '_' + str(idx) + 'ES.tif')
                PIL_image_ES.save(saving_path_ES)


    # ################## Loop through all the patients (EXTRACTION OF THE TTE) ##################
    for patient in os.listdir(directory_tte):

        # Looping through ['2CH_ED', '2CH_ES', '4CH_ED', '4CH_ES']
        for current_tte in type_tte:
            # Preparing images
            image_data, image_gt = prepare_image_data_gt(directory_tte, patient, current_tte)

            if flag_equal_pixel:
                image_data, image_gt = equal_pixel(image_data, image_gt, pixel_spacing=None)

            # Saving Images (GRAY DATA)
            save_medimage_to_PIL(image_data, saving_directory_tte, patient, current_tte, type='gray', resize_dim=resize_dim)
            # Saving Images (GROUND TRUTH DATA)
            save_medimage_to_PIL(image_gt, saving_directory_tte, patient, current_tte, type='gt', resize_dim=resize_dim)

if __name__ == "__main__":
    main()