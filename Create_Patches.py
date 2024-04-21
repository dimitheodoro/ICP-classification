from skimage.util import view_as_blocks

import numpy as np


def create_patches(ALL_PATIENTS_SCANS,desired_block=(128,128,1)):
    Volume=[]
    for patient in range(ALL_PATIENTS_SCANS.shape[0]):
        volume = view_as_blocks(ALL_PATIENTS_SCANS[patient], desired_block)
        volume = np.squeeze(volume)
        volume =  np.transpose(volume,(0,1,3,4,2))
        volume=volume.reshape(-1,128,128,120)
        Volume.append(volume)
    return np.array(Volume)
        

