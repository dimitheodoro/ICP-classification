!pip install pydicom 
import  pydicom as dcm
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom,rotate

path ='test'


def sort_dicoms(path):
    acq_times = ([dcm.read_file(os.path.join(path,slice))[('0008','0032')].value for slice in (os.listdir((path)))])
    scans_times = dict(zip(os.listdir(path),acq_times))
    scans_times = {k: v for k, v in sorted(scans_times.items(), key=lambda item: item[1])}
    sorted_dcm = list(scans_times.keys())
    return sorted_dcm

# sorted_dcm = sort_dicoms(path)

def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dcm.multival.MultiValue: return int(x[0])
    else: return int(x)
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def read_dicom_images(path,sorted_dcm):
    scans = [dcm.read_file(os.path.join(path,slice)) for slice in sorted_dcm]
    slices =  np.array([dcm.read_file(os.path.join(path,slice)).pixel_array for slice in sorted_dcm])
    # slices =np.transpose(slices,(1,2,0))
    window_center , window_width, intercept, slope = get_windowing(scans[0])  
    return  window_image(slices,window_center , window_width, intercept, slope )



sorted_dcms=[]
for patient in os.listdir(path):
    sorted_dcms.append(sort_dicoms(os.path.join(path,patient)))





def process_scan(path,sorted_dcm):
    """Read and resize volume"""
    # Read scan
    volume = read_dicom_images(path,sorted_dcm)
    # Resize width, height and depth

    return volume.astype('float32')

slices=[]


for i,patient in enumerate (os.listdir(path)):

    slices.append(process_scan(os.path.join(path,patient),sorted_dcms[i]))


for i in range(len(slices)):
    slices[i]=np.transpose(slices[i],(1,2,0))

# slices= np.array(slices)
# slices =np.transpose(slices,(0,2,3,1))
# print("final",slices.shape)




##################### resampling         as i want !!!!!!!!!!!!!!

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 256
    desired_height = 256
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # print("---------------",img.shape)
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = zoom(img, (width_factor, height_factor ,depth_factor), order=1)
    return img
 
resampled_slices=[]

for patient in range(len(slices)):
    resampled_slices.append(resize_volume(slices[patient]))
    
    
    

resampled_slices =np.array(resampled_slices)

print("resampled_slices.shape:",resampled_slices.shape)




######################################################## 2D
######################################################## 2D
######################################################## 2D

######################################################## 2D
######################################################## 2D
######################################################## 2D

######################################################## 2D

path ='/content/drive/MyDrive/lung-ct.volume-3d'

def sort_dicoms(path):
    acq_times = ([dcm.read_file(os.path.join(path,slice))[('0008','0032')].value for slice in (os.listdir((path)))])
    scans_times = dict(zip(os.listdir(path),acq_times))
    scans_times = {k: v for k, v in sorted(scans_times.items(), key=lambda item: item[1])}
    sorted_dcm = list(scans_times.keys())
    return sorted_dcm

sorted_dcm = sort_dicoms(path)

def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dcm.multival.MultiValue: return int(x[0])
    else: return int(x)
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def read_dicom_images(path,sorted_dcm):
    scans = [dcm.read_file(os.path.join(path,slice)) for slice in sorted_dcm]
    slices =  np.array([dcm.read_file(os.path.join(path,slice)).pixel_array for slice in sorted_dcm])
    # slices =np.transpose(slices,(1,2,0))
    window_center , window_width, intercept, slope = get_windowing(scans[0])  
    return  window_image(slices,window_center , window_width, intercept, slope )



def process_scan(path,sorted_dcm):
    """Read and resize volume"""
    # Read scan
    volume = read_dicom_images(path,sorted_dcm)
    # Resize width, height and depth

    return volume.astype('float32')



slices =process_scan(path,sorted_dcm)
slices =np.transpose(slices,(1,2,0))

########################################## resampling as i want ###############################################

slices.shape

from scipy.ndimage import zoom,rotate

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # print("---------------",img.shape)
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = zoom(img, (width_factor, height_factor ,depth_factor), order=1)
    return img
 

    

resampled_slices = resize_volume(slices)

print("resampled_slices.shape:",resampled_slices.shape)





########################################## resampling ###############################################
def get_spacing(path):

    pixel_spacing = [dcm.read_file(os.path.join(path,slice)).PixelSpacing for slice in (os.listdir((path)))][:1]
    slice_thickness = [dcm.read_file(os.path.join(path,slice)).SliceThickness for slice in (os.listdir((path)))][:1]
    
    return  pixel_spacing,slice_thickness
                          
pixel_spacing,slice_thickness = get_spacing(path)
# print(pixel_spacing,slice_thickness)

def resample(image,pixel_spacing, slice_thickness , new_spacing=[1,1,1]):

    spacing = np.array([slice_thickness[0],pixel_spacing[0][0],pixel_spacing[0][1]])
    resize_factor = spacing / np.array(new_spacing)  
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor    
    print("new_spacing:",new_spacing)
    image = zoom(image, real_resize_factor, mode='nearest')
  
    
    return image

resampled_slices=resample (slices,pixel_spacing, slice_thickness)

resampled_slices.shape
