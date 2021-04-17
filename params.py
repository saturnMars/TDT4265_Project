from pathlib import Path
import albumentations as A

#epochs
EPOCHS =  1#50

#learning rate
LEARN_RATE = 0.01

# Resolution of the image - Watch out Memory usage (Naive)
# (1, 1.5 or 2, 3, 4) 
scale = 1 

MODEL_PATH = './models/'

FILE_NAME = None

LOAD = False

DATA_PARAMS = {
    'batch_size' : 8,
    'base_path' : Path('data'),
    'dataset' : 'extracted_CAMUS',
    'image_resolution' : int(384 * scale)
}

# The order is relevant here, so be careful when you put something
PREP_STEPS = ['GaussBlur'
            #'MedianFilter',
            #'bright',
            #'EDGE_ENHANCE',
            #'MedianFilter'
            #'Sharp'
            #'MaxFilter'
            ]

TRAIN_TRANSFORMS = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])