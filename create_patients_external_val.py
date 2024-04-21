import pydicom as dcm
import os
import numpy as np
import cv2
from scipy import ndimage
#import matplotlib.pyplot as plt
from segment_brain import segment
from tqdm import tqdm
import re
from CT_DATASET_module_with_Classes_rescale import *


PATH_WITH_ALL_SCANS4 = '/home/theodoropoulos/PhD/Data/NORMAL'
PATH_WITH_ALL_SCANS2 = '/home/theodoropoulos/PhD/Data/WIDE_RANGE'  # New folder path
PATH_WITH_ALL_SCANS3 = '/home/theodoropoulos/PhD/Data/MORE_OR_EQ_'  # New folder path
PATH_WITH_ALL_SCANS = '/home/theodoropoulos/PhD/Data/UNDER15_NORMAL'

desired_volume_dims_after_resampling = (120, 512, 512)
#desired_volume_dims_after_resampling = (64, 128, 128)
THRESHOLD = 20
all_patients = CT_DATASET(PATH_WITH_ALL_SCANS, desired_volume_dims_after_resampling)

patients = []
# Loop over files in the first folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS)[:124]):
    patients.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS, patient), patient))

# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS2)[:38]):
    patients.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS2, patient), patient))

# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS3)[:87]):
    patients.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS3, patient), patient))

# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS4)[:270]):
    patients.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS4, patient), patient))


patients = np.array(patients)
print(patients.shape)

np.save("/home/theodoropoulos/PhD/Data/patients(519_120_512_512)",patients)

patients2 = []
# Loop over files in the first folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS)[124::]):
    patients2.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS, patient), patient))

# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS2)[38::]):
    patients2.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS2, patient), patient))

# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS3)[87::]):
    patients2.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS3, patient), patient))

# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS4)[270::]):
    patients2.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS4, patient), patient))


patients2 = np.array(patients2)
print(patients2.shape)

np.save("/home/theodoropoulos/PhD/Data/patients(59_120_512_512)_external_val",patients2)