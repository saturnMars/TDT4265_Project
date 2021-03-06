import numpy as np
import pandas as pd

import matplotlib as mp
import matplotlib.pyplot as plt

import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import cv2

from src.utils import *

def preprocessing(image, pre_processing_steps, **kwargs):

    image_pre = image.copy()
    bright = kwargs.get('brightness', 1)
    bil_sig_col = kwargs.get('sigmaColor', 15)
    bil_sig_space = kwargs.get('sigmaSpace', 15)
    bil_ker_size = kwargs.get('bilatKernelSize', 15)

    for step in pre_processing_steps:
        if step == 'GaussBlur':
            image_pre = image_pre.filter(ImageFilter.GaussianBlur)
        elif step == 'MedianFilter':
            image_pre = image_pre.filter(ImageFilter.MedianFilter)
        elif step == 'ModeFilter':
            image_pre = image_pre.filter(ImageFilter.ModeFilter)
        elif step == 'MaxFilter':
            image_pre = image_pre.filter(ImageFilter.MaxFilter)
        elif step == 'Sharp':
            image_pre = image_pre.filter(ImageFilter.SHARPEN)
        elif step == 'EDGE_ENHANCE':
            image_pre = image_pre.filter(ImageFilter.EDGE_ENHANCE)
        elif step == 'EDGE_ENHANCE_MORE':
            image_pre = image_pre.filter(ImageFilter.EDGE_ENHANCE_MORE)
        elif step == 'FIND_EDGES':
            image_pre = image_pre.filter(ImageFilter.FIND_EDGES)
        elif step == 'Bright':
            enhancer = ImageEnhance.Brightness(image_pre)
            image_pre = enhancer.enhance(bright)
        elif step == 'BilateralSmooth':
            bilateral = cv2.bilateralFilter(np.array(image_pre), bil_ker_size, bil_sig_col, bil_sig_space)
            image_pre = Image.fromarray(np.uint8(bilateral)).convert('L')
        else:
            raise Exception('Unknown preprocessing step')

    return image_pre

if __name__ == "__main__":

    directory_main = Path('data')
    CAMUS_path = os.path.join(directory_main, 'extracted_CAMUS', 'train_gray')
    TEE_path = os.path.join(directory_main, 'extracted_TEE', 'train_gray')

    test_image = 'gray_patient0001_2CH_ED.tif'
    image = Image.open(os.path.join(CAMUS_path, test_image))
    image.show(title='original')
    # Additional parameters
    dict = {'brightness': 1.5}
    # The order is relevant here, so be careful when you put something
    pre_processing_steps = [#'GaussBlur',
                            'BilateralSmooth'
                            #'MedianFilter',
                            #'bright',
                            #'EDGE_ENHANCE',
                            #'MedianFilter'
                            #'Sharp'
                            #'MaxFilter'
                            ]

    image_pre = preprocessing(image, pre_processing_steps, **dict)
    image_pre.show(title='preprocessed')