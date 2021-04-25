import torch
from torch.utils.data import Dataset, DataLoader, sampler

from src.preprocessing import preprocessing

from random import sample
import random
from pathlib import Path
import numpy as np
from PIL import Image

class DatasetLoaderNoGT(Dataset):
    def __init__(self, gray_files, pytorch = True, prep_steps=None, image_resolution=384):
        super().__init__()
        
        self.files = gray_files
        
        self.pytorch = pytorch
        self.prep_steps = prep_steps
        self.resolution = image_resolution
        
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)

    def open_as_array(self, idx, invert=False):
        # Open ultrasound data
        PIL_image = Image.open(self.files[idx]).resize((self.resolution, self.resolution))

        # Pre_processing steps
        if self.prep_steps:
            PIL_image = preprocessing(PIL_image, self.prep_steps)

        raw_us = np.stack([np.array(PIL_image)], axis=2)
    
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    
    def __getitem__(self, idx):
        x = self.open_as_array(idx, invert=self.pytorch).astype('float32')
        
        return x
    
#load data from a folder
class DatasetLoader(DatasetLoaderNoGT):
    def __init__(self, gray_files, gt_dir, pytorch=True, prep_steps=None, transform=None, image_resolution=384):
        super().__init__(gray_files, pytorch, prep_steps, image_resolution)
        
        self.transform = transform
        
        # Loop through the files in red folder and combine, into a dictionary, the other band
        self.files = gray_files
        self.gt_files = [gt_dir/gray_file.name.replace('gray', 'gt') for gray_file in self.files]
        
    def open_mask(self, idx, add_dims=False):
        #open mask file
        raw_mask = np.array(Image.open(self.gt_files[idx]).resize((self.resolution, self.resolution)))
        #raw_mask = np.where(raw_mask>100, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        x = super().__getitem__(idx)
        #get the image and mask as arrays
        y = self.open_mask(idx, add_dims=False)
        
        if self.transform is not None:
            transformed = self.transform(image=x, mask=y)
            x = transformed['image']
            y = transformed['mask']

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.torch.int64)
        
        return x, y
    
    def get_as_pil(self, idx):
        #get an image for visualization
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')


    
def train_test_val_split(base_path, dataset, database_size):
    train_folder = 'train_gray'
    test_folder = 'test_gray'
    
    # Load train data
    train_files = [f for f in Path.joinpath(base_path, dataset, train_folder).iterdir() if not f.is_dir()]
    train_files = sample(train_files, database_size)

    # Split train-validation data
    val_size = int(0.1 * len(train_files))
    
    val_file_ids = sample(range(len(train_files)), val_size)
    train_file_ids = [i for i in range(len(train_files)) if i not in val_file_ids]

    train_df = [train_files[i] for i in train_file_ids]
    val_df = [train_files[i] for i in val_file_ids]

    # Load test data
    test_df = [f for f in Path.joinpath(base_path, dataset, test_folder).iterdir() if not f.is_dir()]

    return train_df, test_df, val_df

def load_train_test_val(data_params, prep_steps=None, train_transform=None):
    base_path = data_params['base_path']
    dataset = data_params['dataset']
    image_resolution = data_params['image_resolution']
    batch_size = data_params['batch_size']
    database_size = data_params['database_size']
    
    print(f"DATASET: {dataset}")
    train_files, test_files, val_files = train_test_val_split(base_path, dataset, database_size = database_size)

    # Dataset loaders
    train_dataset = DatasetLoader(train_files,
                                  Path.joinpath(base_path, dataset, 'train_gt'),
                                  prep_steps = prep_steps,
                                  transform = train_transform,
                                  image_resolution = image_resolution)

    valid_dataset = DatasetLoader(val_files,
                                  Path.joinpath(base_path, dataset, 'train_gt'),
                                  prep_steps = prep_steps,
                                  image_resolution = image_resolution)

    test_dataset = DatasetLoader(test_files,
                                 Path.joinpath(base_path, dataset, 'test_gt'),
                                  prep_steps = prep_steps,
                                  image_resolution = image_resolution)

    # Data Loaders
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print(f"\nItems loaded: {len(train_dataset)+len(test_dataset)+len(valid_dataset)}" 
          f"[train: {len(train_dataset)}, valid: {len(valid_dataset)} test: {len(test_dataset)}]")

    # Visualize shape of raw and ground true images
    xb, yb = next(iter(train_data))
    print(f"RAW IMAGES: {xb.shape}\n GT IMAGES: {yb.shape}\n")
    
    return train_data, test_data, valid_data

def load_tee(base_path, batch_size, prep_steps):
    dataset = 'extracted_TEE'
    files = [f for f in Path.joinpath(base_path, dataset, 'test_gray').iterdir() if not f.is_dir()]
 
    dataset = DatasetLoader(files, Path.joinpath(base_path, dataset, 'test_gt'), prep_steps = prep_steps)
    print("TEE IMAGES:", len(dataset))
  
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
    xb, yb = next(iter(data))
    print(f"RAW TEE IMAGES: {xb.shape}\n GT IMAGES: {yb.shape}\n")
    
    return data

def load_tee_no_gt(base_path, prep_steps):
    dataset = 'extracted_TEE'
    files = [f for f in Path.joinpath(base_path, dataset, 'final_test').iterdir() if not f.is_dir()]
    
    dataset = DatasetLoaderNoGT(files, prep_steps=prep_steps)
    print('TEE IMAGES wo. gt: ', len(dataset))
    
    return dataset