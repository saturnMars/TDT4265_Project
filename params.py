from pathlib import Path
import albumentations as A

#epochs
EPOCHS =  50

#learning rate
LEARN_RATE = 0.01

# Resolution of the image - Watch out Memory usage (Naive)
scale = 1 

# Scale database
db_scale = 1

MODEL_PATH = './models/'

FILE_NAME = 'improved_unet.pt'

LOAD = False

DATA_PARAMS = {
    'batch_size' : 8,
    'base_path' : Path('data'),
    'dataset' : 'extracted_CAMUS',
    'image_resolution' : int(384 * scale),
    'database_size' : int(1600 * db_scale)
}

# The order is relevant here, so be careful when you put something
PREP_STEPS = ['GaussBlur',
            #'MedianFilter',
            #'bright',
            #'EDGE_ENHANCE',
            #'MedianFilter'
            'Sharp'
            #'MaxFilter'
            ]

TRAIN_TRANSFORMS = A.Compose([
    #A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
])
