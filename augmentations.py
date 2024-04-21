import tensorflow as tf
import numpy as np
from scipy import ndimage
import random
import cv2
from scipy.ndimage import gaussian_filter

#@tf.function
def CT_augmentations(volume):
    """Rotate the volume by a few degrees"""
    
    def scipy_rotate(volume):
        
        if tf.reduce_all(tf.equal(volume, 0)).numpy():
            return volume.astype('float64')
        else:
             # Define some rotation angles
            angles = [-20, -10, -5, 5, 10, 20]
            # Pick an angle at random
            angle = random.choice(angles)
            # Rotate volume
            volume = ndimage.rotate(volume, angle, reshape=False)
            volume[volume < 0] = 0
            volume[volume > 1] = 1
            # print(volume)
            # print(volume)
            return volume.astype('float64')
            


    def crop_image(image):
        # Get original image dimensions
        original_height, original_width = image.shape[:2]
        # Generate random coordinates for cropping
        crop_top = random.randint(0, original_height )
        crop_left = random.randint(0, original_width )
        crop_height = random.randint(1, original_height - crop_top)
        crop_width = random.randint(1, original_width - crop_left)
        # Calculate bottom and right coordinates of the cropping region
        crop_bottom = min(original_height, crop_top + crop_height)
        crop_right = min(original_width, crop_left + crop_width)
        # Crop the image
        cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]
        # Pad cropped region if necessary to match original shape
        cropped_image_padded = np.zeros((original_height, original_width, image.shape[2]), dtype=image.dtype)
        cropped_image_padded[crop_top:crop_bottom, crop_left:crop_right] = cropped_image
        return cropped_image_padded
    
    def gaussianfilter(volume):
        if tf.reduce_all(tf.equal(volume, 0)).numpy():
            return volume.astype('float64')
        else:
            # print(gaussian_filter(volume, sigma=1))
            return gaussian_filter(volume, sigma=1).astype('float64')
            

    def create_probe_artifact(volume):
        def for_one_slice(image):
            img_copy = image.copy()
            THRESHOLD = 0.1
            thresh = img_copy*255>THRESHOLD
            # Find contours of bright regions
            contours, _ = cv2.findContours(thresh.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw star-like pattern on the image
            for contour in contours:
                # Calculate the centroid of each contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"]*2 / M["m00"]*0.02) #int(M["m10"] / M["m00"])
                    cY = int(M["m01"]*1.6 / M["m00"]*0.2)
                else:
                    cX, cY = 50, 50
                # Draw lines from centroid to contour points
                for point in contour:
                    x, y = point[0]
                    cv2.line(img_copy, (cX, cY), (x, y), (255, 255, 255), 1)
            distorted = img_copy*image
            return distorted
        
        
        volume_with_artifact= []
        for slice in range(volume.shape[-1]):
            volume_with_artifact.append(for_one_slice(volume[:,:,slice]))
        volume_with_artifact = np.array(volume_with_artifact)
        volume_with_artifact = np.transpose(volume_with_artifact,(1,2,0))
        return volume_with_artifact

    def clipped_zoom(img, zoom_factor=1.2, **kwargs):
        h, w = img.shape[:2]
        # For multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
        # Zooming out
        if zoom_factor < 1:
            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)
        # Zooming in
        elif zoom_factor > 1:
            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            # trim_top = ((out.shape[0] - h) // 2)
            # trim_left = ((out.shape[1] - w) // 2)
            # out = out[trim_top:trim_top+h, trim_left:trim_left+w]
        # If zoom_factor == 1, just return the input array
        else:
            out = img
        return out

    def apply_augmentation(volume):
        volume_augmented = volume

        # Apply augmentations with a certain probability
        # if random.random() < 0.5:
        #     print("create_probe_artifact")
        #     volume_augmented = create_probe_artifact(volume_augmented)  
        if random.random() < 0.5:
           # print("scipy_rotate")
            volume_augmented = scipy_rotate(volume_augmented)
        # if random.random() < 0.5:
        #     print("crop_image")
        #     volume_augmented = crop_image(volume_augmented)
        # if random.random() < 0.5:
        #     print("clipped_zoom")
        #     volume_augmented = clipped_zoom(volume_augmented,1.5)
        if random.random() < 0.5:
            #print("blur")
            volume_augmented = gaussianfilter(volume_augmented)

        return volume_augmented

    def min_max_normalize(array):
        """
        Normalize a SymbolicTensor to the range [0, 1] using min-max normalization.
        
        Parameters:
        array (SymbolicTensor): The input SymbolicTensor to be normalized.
        
        Returns:
        SymbolicTensor: The normalized SymbolicTensor.
        """
        # Convert SymbolicTensor to a TensorFlow tensor
        array_tf = tf.convert_to_tensor(array)
        
        # Calculate min and max values
        min_val = tf.reduce_min(array_tf)
        max_val = tf.reduce_max(array_tf)

        if min_val==max_val:
            max_val += 1e-10
            # Apply min-max normalization
            normalized_array = (array_tf - min_val) / (max_val - min_val)
            return normalized_array
        elif min_val!=max_val:
            normalized_array = (array_tf - min_val) / (max_val - min_val)
            return normalized_array
    

    def set_zero_to_image(volume):
        counts, bins = tf.numpy_function(np.histogram, [volume], ['float64', 'float64'])
        bin_max_index = tf.argmax(counts)
        mask = tf.less_equal(volume, bins[bin_max_index + 1])
        masked_volume = tf.where(mask, tf.zeros_like(volume), volume)
        return masked_volume


    # def apply_augmentation(volume):
    #     augmentations_list = [scipy_rotate,crop_image,gaussianfilter,clipped_zoom,create_probe_artifact]
    #     selected =  np.random.choice(augmentations_list)
    #     print(selected)
    #     return selected(volume)



    volume = tf.numpy_function(apply_augmentation, [volume], 'float64')
    volume = tf.numpy_function(min_max_normalize, [volume], 'float64')
    # volume = tf.numpy_function(set_zero_to_image, [volume], 'float64')
    return volume
      
