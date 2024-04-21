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


PATH_WITH_ALL_SCANS = '/home/theodoropoulos/Desktop/Datasets/UNDER15_NORMAL' #124
PATH_WITH_ALL_SCANS2 = '/home/theodoropoulos/Desktop/Datasets/WIDE_RANGE'  #87
PATH_WITH_ALL_SCANS3 = '/home/theodoropoulos/Desktop/Datasets/MORE_OR_EQ_'  #38 
PATH_WITH_ALL_SCANS4 = '/home/theodoropoulos/Desktop/Datasets/NORMAL' #270

PATH_TO_SAVE ='/home/theodoropoulos/Desktop'

desired_volume_dims_after_resampling = (120, 512, 512)
THRESHOLD = 20
all_patients = CT_DATASET(PATH_WITH_ALL_SCANS, desired_volume_dims_after_resampling)

######################################################### 124  UNDER15_NORMAL ##########################################################  

patients_under_15 = []
# Loop over files in the first folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS)[:60]):
    patients_under_15.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS, patient), patient))
    
patients_under_15 = np.array(patients_under_15)
np.save(os.path.join(PATH_TO_SAVE,'patients_under_15_1'),patients_under_15)
del patients_under_15

patients_under_15 = []
# Loop over files in the first folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS)[60:124]):
    patients_under_15.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS, patient), patient))
    
patients_under_15 = np.array(patients_under_15)
np.save(os.path.join(PATH_TO_SAVE,'patients_under_15_2'),patients_under_15)
del patients_under_15

##############################################   87   WIDE_RANGE #############################################

patients_wide_range = []
# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS2)[:60]):
    patients_wide_range.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS2, patient), patient))
  
patients_wide_range = np.array(patients_wide_range)
np.save(os.path.join(PATH_TO_SAVE,'patients_wide_range_1'),patients_wide_range)
del patients_wide_range  
    
patients_wide_range = []
# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS2)[60:87]):
    patients_wide_range.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS2, patient), patient))
  
patients_wide_range = np.array(patients_wide_range)
np.save(os.path.join(PATH_TO_SAVE,'patients_wide_range_2'),patients_wide_range)
del patients_wide_range  
    

#################################################    38  MORE_OR_EQ_ ######################################


patients_more_or_eq = []
# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS3)[:38]):
    patients_more_or_eq.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS3, patient), patient))
    
patients_more_or_eq = np.array(patients_more_or_eq)
np.save(os.path.join(PATH_TO_SAVE,'patients_more_or_eq'),patients_more_or_eq)
del patients_more_or_eq  
    
    
############################################################ 270 NORMAL ###################################

patients_normal = []
# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS4)[:60]):
    patients_normal.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS4, patient), patient))

patients_normal = np.array(patients_normal)
np.save(os.path.join(PATH_TO_SAVE,'patients_normal_1'),patients_normal)
del patients_normal

patients_normal = []
# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS4)[60:120]):
    patients_normal.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS4, patient), patient))

patients_normal = np.array(patients_normal)
np.save(os.path.join(PATH_TO_SAVE,'patients_normal_2'),patients_normal)
del patients_normal

patients_normal = []
# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS4)[120:180]):
    patients_normal.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS4, patient), patient))

patients_normal = np.array(patients_normal)
np.save(os.path.join(PATH_TO_SAVE,'patients_normal_3'),patients_normal)
del patients_normal

patients_normal = []
# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS4)[180:240]):
    patients_normal.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS4, patient), patient))

patients_normal = np.array(patients_normal)
np.save(os.path.join(PATH_TO_SAVE,'patients_normal_4'),patients_normal)
del patients_normal

patients_normal = []
# Loop over files in the second folder
for patient in tqdm(os.listdir(PATH_WITH_ALL_SCANS4)[240:270]):
    patients_normal.append(all_patients.process_scan(os.path.join(PATH_WITH_ALL_SCANS4, patient), patient))

patients_normal = np.array(patients_normal)
np.save(os.path.join(PATH_TO_SAVE,'patients_normal_5'),patients_normal)
del patients_normal

