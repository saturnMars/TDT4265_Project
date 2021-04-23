import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from medimage import image
import os
import h5py
import cv2
from pathlib import Path
from scipy.signal import find_peaks
import re


def prepare_TTE_image_data_gt(directory_train, patient, type):
    """
    This function is used to extract ES ED pair of images data with its associated ground truth image, from the CAMUS
    dataset (TTE images)
    Parameters
    ----------
    directory_train: String | Path
        Main path where you there are all the patient folders
    patient: String | Path
        Current patient folder
    type: String
        Type of image (['2CH_ED', '2CH_ES','4CH_ED','4CH_ES'])
    Return
    -------
    image_data: medimage.Image
        Image containing the image data
    image_gt: medimage.Image
        Image containing the ground truth data (classes)
    """

    path_data = os.path.join(directory_train, patient, patient + '_' + type +'.mhd')
    path_gt = os.path.join(directory_train, patient, patient + '_' + type + '_gt.mhd')

    # Retrieve the data
    image_data = image(path_data)
    image_gt = image(path_gt)
    return image_data, image_gt


def save_from_PIL_to_tif(PIL_image, saving_directory, patient, image_type, type, resize_dim=None):
    """
    Function used to convert a image from PIL format to a tif format. Before saving it the image will be resized
    in order to have a square image with the given resolution (i.e. resize_dim)

    Parameters
    ----------
    PIL_image: PIL image
        TTE / TEE image
    saving_directory: Path | string
        Directory where to save the image
    patient: string
        String of the patient name
    image_type: string
        Type of image (For TTE: ['2CH_ED', '2CH_ES','4CH_ED','4CH_ES']. For TEE: ['ED','ES'])
    type: string
        String that define if the image is a "gray" or "ground truth" image ('gray', 'gt')
    resize_dim: int
        Int that the decide the final resolution of the image

    """

    # Resize the image considering the highest dimension
    max_dimension = max(PIL_image.size)

    # Make the image's pixels iso-tropic
    PIL_image = PIL_image.resize((max_dimension, max_dimension))

    if resize_dim is not None:
        # Resize to the desired resolution
        PIL_image = PIL_image.resize((resize_dim, resize_dim))
    else:
        pass

    if type == 'gray':
        type_path = 'train_gray'
    elif type == 'gt':
        type_path = 'train_gt'
    elif type == 'test_gray':
        type_path = 'test_gray'
    elif type == 'test_gt':
        type_path = 'test_gt'
    elif type == 'final_test':
        type_path = 'final_test'
    else:
        raise Exception('Unknown data type')

    saving_path = os.path.join(saving_directory, type_path, type + '_' + patient + '_' + image_type + '.tif')
    PIL_image.save(saving_path)
    return


def medimage_to_PIL(medimage):
    """ This function is used to convert a medimage.image object to a PIL image """
    # Converting medimage.image into PIL image
    return Image.fromarray(np.uint8(medimage.imdata.squeeze())).convert('L')


def ndarray_to_PIL(narray_image):
    """ This function is used to convert a ndarray image object to a PIL image """
    # Converting hdf5 into PIL image
    return Image.fromarray(np.uint8(narray_image.squeeze())).convert('L')


def custom_tee_verbose(verbose_index):
    """ This function will give verbose info on the extraction"""
    verbose_index += 1
    if verbose_index % 10 == 0:
        print(f'HDF5 files underwent to extraction: {verbose_index}')
    return verbose_index


def prepare_TEE_to_ED_ES(path_hdf5, heartbeat_duration, user_response):
    """
    This function will extract from the hdf5 file two images corresponding to two cardiac events ED and ES
    Parameters
    ----------
    path_hdf5: String | Path
        Path to the current hdf5 file
    heartbeat_duration: float
        Heartbeat duration in sec
    user_response: str
        Flag used to decide if to extract the ES event from the ECG signal
    Return
    -------
    tissue_all: Dictionary
        Dict containing the numpy.ndarray images (can be only ED or both ED/ES)
    """
    tissue_all = {}
    image_hdf5 = h5py.File(path_hdf5, 'r')

    # Some parameters
    QRS_time = 0.1      # [sec]
    QT_time = 0.400     # [sec] Found in literature

    # Extract the tissue
    tissue_hdf5 = image_hdf5['tissue']
    tissue_data = tissue_hdf5['data']

    # Extract the ECG
    ecg_hdf5 = image_hdf5['ecg']
    ecg_data = ecg_hdf5['ecg_data']
    ecg_times = ecg_hdf5['ecg_times']

    index_list = [i for i in range(ecg_data.shape[0])]
    ecg_data_arr = ecg_data.__getitem__(index_list)
    ecg_times_arr = ecg_times.__getitem__(index_list)

    # Sampling time
    delta_t = ecg_times_arr[1] - ecg_times_arr[0]
    # Sampling frequency
    fs = np.round(1 / delta_t)

    # =========== EXTRACTION OF THE ED EVENT ===========
    # Find the peaks
    max_point = np.max(ecg_data_arr)
    heartbeat_n_samples = fs * heartbeat_duration
    ed_event, _ = find_peaks(ecg_data_arr, height=max_point/2, distance=np.round(heartbeat_n_samples / 2))
    # We need to find the index of the image associated to the time-instance (Sampling time of ECG different of TEE
    # acquisition frequency)
    idx_ed = int(np.round(tissue_data.shape[2] * ed_event[0] / len(index_list)))
    # We need to rotate the image of 180°
    tissue_all['ED'] = np.fliplr(tissue_data[:, :, idx_ed])

    if str.lower(user_response) == 'y':
        # =========== EXTRACTION OF THE ES EVENT ===========
        # Due to high complexity of ES's event identification the user will need to choose one point in the ECG signal

        # #################### Plotting the ECG for ES identification purpose #####################
        plt.plot(ecg_data_arr)
        plt.plot(ed_event, ecg_data_arr[ed_event], "x")
        plt.title('ECG - (Please click on the ES event location)')
        plt.xlabel('# Samples ')
        plt.ylabel('Amplitude [a.u]')
        plt.xlim([index_list[0], index_list[-1]])
        # #########################################################################################

        es_event = plt.ginput(n=3, show_clicks=True)
        plt.close()

        idx_es = int(np.round(tissue_data.shape[2] * es_event[0][0] / len(index_list)))
        # We need to rotate the image of 180°
        tissue_all['ES'] = np.fliplr(tissue_data[:, :, idx_es])

    elif str.lower(user_response) == 'n':
        # The ES event will be extracted in according to Literature measurements
        # https://emedicine.medscape.com/article/2172196-overview
        RT_time = QT_time - QRS_time/2
        es_event = RT_time * fs + ed_event[0]
        idx_es = int(np.round(tissue_data.shape[2] * es_event / len(index_list)))
        tissue_all['ES'] = np.fliplr(tissue_data[:, :, idx_es])

    else:
        raise Exception('Unknown user response')

    return tissue_all



def main():

    # Directory where there is the TRAINING CAMUS DATASET / TEE
    directory_tte = Path('../CAMUS_dataset')
    directory_tee = Path('../TEE/TEE_unlabeled')
    directory_tee_labeled = ('../TEE/TEE_labeled')
    # Directory where i would like to store the images extracted from the CAMUS and TEE
    saving_directory_tte = Path('../data/extracted_CAMUS')
    saving_directory_tee = Path('../data/extracted_TEE')

    # PARAMS
    resize_dim = None
    # Hearbeat duration according to the literature
    heartbeat_duration = 1    # [sec]   # We assume 60 bpm for semplicity
    # Type of image that we want to extract for TTE dataset
    type_tte = ['2CH_ED', '2CH_ES', '4CH_ED', '4CH_ES']

    # ################## Loop through all the patients (EXTRACTION OF THE TEE - WITHOUT GT) ##################

    verbose_index = 0
    user_response = input('Do you want to extract (it will be manual) the ES EVENTS for the TEE dataset? [y/n]')
    if str.lower(user_response) == 'n':
        print("The ES event will be extracted by using an empirical QT' duration")

    print('.. Extraction of TEE images (without the GT data)')

    for patient in os.listdir(directory_tee):

        # Looping through all the hdf5 file for each patient
        for current_hdf5 in os.listdir(os.path.join(directory_tee, patient)):

            # Just for verbose info
            verbose_index = custom_tee_verbose(verbose_index)

            # Path to the current .hdf5 file
            path_hdf5 = os.path.join(directory_tee, patient, current_hdf5)

            # Extraction of the image in correspondence of the two events
            tissue_all = prepare_TEE_to_ED_ES(path_hdf5, heartbeat_duration, user_response)

            # Converting the images to PIL
            tee_ed = tissue_all['ED']
            PIL_tte_ed = ndarray_to_PIL(tee_ed)

            # Saving Images (GRAY DATA ..)
            save_from_PIL_to_tif(PIL_tte_ed, saving_directory_tee, current_hdf5, 'ED', 'final_test', resize_dim=resize_dim)

            if 'ES' in tissue_all:

                tee_es = tissue_all['ES']
                # Converting the images to PIL
                PIL_tte_es = ndarray_to_PIL(tee_es)
                # Saving Images (GRAY DATA ..)
                save_from_PIL_to_tif(PIL_tte_es, saving_directory_tee, current_hdf5, 'ES', 'final_test', resize_dim=resize_dim)

    # ################## Loop through all the patients (EXTRACTION OF THE TEE - WITH GT) ##################
    print('.. Extraction of TEE images (with the GT data)')
    path_tee_labeled_gray = os.listdir(os.path.join(directory_tee_labeled, 'train_gray'))
    path_tee_labeled_gt = os.listdir(os.path.join(directory_tee_labeled, 'train_gt'))

    # Loop trough the given folder
    for tee_gray_label, tee_gt_label in zip(path_tee_labeled_gray, path_tee_labeled_gt):
        # Load image into a PIL format
        PIL_tee_gray_arr = cv2.imread(os.path.join(directory_tee_labeled, 'train_gray', tee_gray_label))
        PIL_tee_gt_arr = cv2.imread(os.path.join(directory_tee_labeled, 'train_gt', tee_gt_label))

        # Cropp the image to remove the black background
        height, width = np.max(PIL_tee_gray_arr, axis=1), np.max(PIL_tee_gray_arr, axis=0)
        idx_h = list(np.where(height > 0)[0])
        idx_w = list(np.where(width > 0)[0])

        # Cropping the images
        cropped_gray = PIL_tee_gray_arr[idx_h[0]:idx_h[-1], idx_w[0]:idx_w[-1]]
        cropped_gt = PIL_tee_gt_arr[idx_h[0]:idx_h[-1], idx_w[0]:idx_w[-1]]

        # Here we don't have class as 1,2,3,4 but we have 127 and 256
        cropped_gt[cropped_gt == 127] = 1
        cropped_gt[cropped_gt == 255] = 2
        # cropped_gt[(cropped_gt != 255) & (cropped_gt != 127)] = 0

        cropped_gray_PIL = ndarray_to_PIL(np.fliplr(cropped_gray))
        cropped_gt_PIL = ndarray_to_PIL(np.fliplr(cropped_gt))



        # Extract the string
        patient_gray = tee_gray_label.split('.jpg')[0]
        patient_gt = tee_gt_label.split('.tif')[0]
        save_from_PIL_to_tif(cropped_gray_PIL, saving_directory_tee, patient_gray, '', 'test_gray', resize_dim=resize_dim)
        save_from_PIL_to_tif(cropped_gt_PIL, saving_directory_tee, patient_gt, '', 'test_gt', resize_dim=resize_dim)

    # ################## Loop through all the patients (EXTRACTION OF THE TTE - CAMUS) ##################
    for patient in os.listdir(directory_tte):
        pat_num = int(re.search(r'\d+', patient).group())
        # Looping through ['2CH_ED', '2CH_ES', '4CH_ED', '4CH_ES']
        for current_tte in type_tte:

            # Preparing images
            image_data_tte, image_gt_tte = prepare_TTE_image_data_gt(directory_tte, patient, current_tte)

            # Converting the images to PIL
            image_data_tte_PIL = medimage_to_PIL(image_data_tte)
            image_gt_tte_PIL = medimage_to_PIL(image_gt_tte)

            # Saving Images (GRAY DATA & GROUND TRUTH)
            if pat_num < 401:
                save_from_PIL_to_tif(image_data_tte_PIL, saving_directory_tte, patient, current_tte, 'gray', resize_dim=resize_dim)
                save_from_PIL_to_tif(image_gt_tte_PIL, saving_directory_tte, patient, current_tte, 'gt', resize_dim=resize_dim)
            else:
                save_from_PIL_to_tif(image_data_tte_PIL, saving_directory_tte, patient, current_tte, 'test_gray', resize_dim=resize_dim)
                save_from_PIL_to_tif(image_gt_tte_PIL, saving_directory_tte, patient, current_tte, 'test_gt', resize_dim=resize_dim)

if __name__ == "__main__":
    main()