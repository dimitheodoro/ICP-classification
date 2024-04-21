import pydicom as dcm
import os
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from segment_brain import segment
from tqdm import tqdm
import re
from segment_brain import segment_all_patients_slices
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from My_model2 import MultipleInputsModel_TURBO,CNN_TURBO
# from CT_DATASET_module_with_Classes_rescale import *
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
import random
from augmentations import CT_augmentations
from sklearn.utils.class_weight import compute_class_weight



strategy = tf.distribute.MirroredStrategy()

patients = np.load("/home/theodoropoulos/Desktop/Datasets/patients(578_120_128_128).npy",allow_pickle=True)

X = np.array([patients[i]['volume']  for i in range(len(patients)) ])
X =np.transpose(X,(0,2,3,1))
X = segment_all_patients_slices(X)
print(X.shape)
y = np.array (  [patients[i]['Class']  for i in range(len(patients)) ]).astype('int32')


X_train,  X_val,  y_train, y_val,   = train_test_split( X, y,test_size=0.3)

X_val,  X_test,  y_val, y_test,   = train_test_split( X_val, y_val,test_size=0.1)


model =CNN_TURBO(input_shape=(128,128,120))
####################

def train_preprocessing(volume,y):
# def train_preprocessing(volume,y):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    # volume = rotate(volume)
    volume = CT_augmentations(volume)
    
    # volume = tf.expand_dims(volume, axis=3)
    print(volume.dtype)
    return (volume),y
    # return volume,y

def validation_preprocessing(volume,y):
# def test_preprocessing(volume,y):

    # volume = tf.expand_dims(volume, axis=3)
    return (volume),y
    # return volume,y

def test_preprocessing(volume,y):
# def test_preprocessing(volume,y):

    # volume = tf.expand_dims(volume, axis=3)
    return (volume),y
    # return volume,y

####################################

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((X_train,y_train))

validation_loader = tf.data.Dataset.from_tensor_slices((X_val, y_val))

test_loader = tf.data.Dataset.from_tensor_slices((X_test,y_test))    
   
batch_size = 8

train_dataset = (
    train_loader.shuffle(len(X_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(1)
)

validation_dataset = (
    validation_loader.shuffle(len(X_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(1)
)

test_dataset = (
    test_loader.shuffle(len(X_test))
    .map(test_preprocessing)
    .batch(batch_size)
    .prefetch(1)
)

########################################

class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weight = {0:class_weight[0], 1:class_weight[1]}

####################
# Create and compile the distributed model
with strategy.scope():
    distributed_model = model

epochs = 30
history = distributed_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    class_weight=class_weight,
    # callbacks=[checkpoint_cb, early_stopping_cb],
)

########################



# Plot training and validation loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
#plt.show()
plt.savefig("/home/theodoropoulos/Desktop/Results/no_patches(120,128,128).png")
distributed_model.save("/home/theodoropoulos/Desktop/Results/full_image_model.h5")

# Validate your model on the validation set
validation_score = distributed_model.evaluate(validation_dataset)
print("Validation Score:",dict(zip(distributed_model.metrics_names, validation_score)))

# Evaluate your final model on the test set
test_score = distributed_model.evaluate(test_dataset)
print("Test Score:", dict(zip(distributed_model.metrics_names, test_score)))
