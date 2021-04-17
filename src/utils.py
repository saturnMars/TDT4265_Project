import torch
import numpy as np

def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements
    
def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    #p = F.softmax(predb[idx], 0)
    #return p.argmax(0).cpu()
    return predb[idx].argmax(0).cpu()

def pre_processing_verbose(pre_processing_steps):
    if len(pre_processing_steps) > 1:
        print(f'The dataset will be undergoes the following pre-processing steps: ')
        for step in pre_processing_steps:
            print(f'{step}')
    return
