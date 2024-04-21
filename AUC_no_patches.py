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
from My_model import MultipleInputsModel
# from CT_DATASET_module_with_Classes_rescale import *
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
import random
from augmentations import CT_augmentations
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#os.environ["CUDA_VISIBLE_DEVICES"] ="1"
#num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
# Set environment variable to specify which GPUs to use
#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(num_gpus)])
# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()




patients = np.load("/home/theodoropoulos/PhD/Data/patients(578_120_128_128).npy",allow_pickle=True)
#desired_volume_dims_after_resampling = (120,256,256)

X = np.array([patients[i]['volume']  for i in range(len(patients)) ])
X =np.transpose(X,(0,2,3,1))
X = segment_all_patients_slices(X)
print(X.shape)
y = np.array (  [patients[i]['Class']  for i in range(len(patients)) ]).astype('float32')

#######################
labels_sex = np.array([patients[i]['sex']  for i in range(len(patients)) ])
le = LabelEncoder()
le.fit(labels_sex)
labels_sex_transf = le.transform(labels_sex)

######################

labels_age = np.array([patients[i]['age']  for i in range(len(patients)) ])
labels_age_categ = []
for age in labels_age:
    if age=='NA':
        labels_age_categ.append('NA')
    elif int(age)<30:
        labels_age_categ.append('Adult')
    elif int(age)>=30 and int(age)<60:
        labels_age_categ.append('Middle')
    else:
        labels_age_categ.append('Old')

labels_age_categ =np.array(labels_age_categ)

le = LabelEncoder()
le.fit(labels_age_categ)
labels_age_transf = le.transform(labels_age_categ)
####################
labels_GCS = np.array([patients[i]['Glasgow Coma Scale']  for i in range(len(patients)) ])
labels_GCS_categ = []
for GCS in labels_GCS:
    if GCS=='NA':
        labels_GCS_categ.append('NA')
    elif int(GCS)<=8:
        labels_GCS_categ.append('HIGH')
    else:
        labels_GCS_categ.append('LOW')

labels_GCS_categ =np.array(labels_GCS_categ)

le = LabelEncoder()
le.fit(labels_GCS_categ)
labels_GCS_transf = le.transform(labels_GCS_categ)

##############################

X_train,  X_val,  y_train, y_val,   labels_sex_train, labels_sex_val,  labels_age_train, labels_age_val ,label_GCS_train, label_GCS_val= train_test_split( X, y,
                                                                                                                                                                labels_sex_transf,
                                                                                                                                                                labels_age_transf,
                                                                                                                                                                labels_GCS_transf,
                                                                                                                                                                test_size=0.3)


X_val,  X_test,  y_val, y_test,   labels_sex_val, labels_sex_test,  labels_age_val, labels_age_test ,label_GCS_val, label_GCS_test= train_test_split( X_val, y_val,
                                                                                                                                                                labels_sex_val,
                                                                                                                                                                labels_age_val,
                                                                                                                                                                label_GCS_val,
                                                                                                                                                                test_size=0.1)




model =MultipleInputsModel(input_shape=(128,128,120),sex_label_shape=(1,),age_label_shape=(1,),GCS_label_shape=(1,),
                           age_num_classes=len(np.unique(labels_age_transf)),
                           sex_num_classes=len(np.unique(labels_sex_transf)),
                           GCS_num_classes=len(np.unique(labels_GCS_transf)),
                           )
####################

def train_preprocessing(volume, labels_sex,labels_age,labels_gcs,y):
# def train_preprocessing(volume,y):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    # volume = rotate(volume)
    volume= CT_augmentations(volume)
    # volume_aug = CT_augmentations(volume[:60])
    # volume_not_aug = volume[60::]
    # volume = tf.concat((volume_aug,volume_not_aug),axis=0)
    
    # volume = tf.expand_dims(volume, axis=3)
    print(volume.dtype)
    return (volume, labels_sex,labels_age,labels_gcs),y
    # return volume,y

def validation_preprocessing(volume, labels_sex,labels_age,labels_gcs,y):
# def test_preprocessing(volume,y):

    # volume = tf.expand_dims(volume, axis=3)
    return (volume, labels_sex,labels_age,labels_gcs),y
    # return volume,y

def test_preprocessing(volume, labels_sex,labels_age,labels_gcs,y):
# def test_preprocessing(volume,y):

    # volume = tf.expand_dims(volume, axis=3)
    return (volume, labels_sex,labels_age,labels_gcs),y
    # return volume,y

####################################

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((X_train,    
                                                     labels_sex_train, 
                                                     labels_age_train, 
                                                     label_GCS_train, 
                                                     y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((X_val,    
                                                     labels_sex_val, 
                                                     labels_age_val, 
                                                     label_GCS_val, 
                                                     y_val))

test_loader = tf.data.Dataset.from_tensor_slices((X_test,    
                                                     labels_sex_test, 
                                                     labels_age_test, 
                                                     label_GCS_test, 
                                                     y_test))


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
print(class_weight)
####################
# Create and compile the distributed model
with strategy.scope():
    distributed_model = model

epochs = 300


checkpoint_filepath = "/home/theodoropoulos/PhD/Results/model.keras"

model_checkpoint_callback =tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history = distributed_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    class_weight=class_weight,
    callbacks=[model_checkpoint_callback],
)

########################



# Plot training and validation loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['auc'], label='Training Accuracy')
plt.plot(history.history['val_auc'], label='Validation Accuracy')
plt.title('Training and Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

plt.tight_layout()
#plt.show()
# plt.savefig("/home/theodoropoulos/PhD/Results/AUC(120,128,128).png")
distributed_model.save("/home/theodoropoulos/PhD/Results/final_model.keras")

# Validate your model on the validation set
validation_score = distributed_model.evaluate(validation_dataset)
print("Validation Score:",dict(zip(distributed_model.metrics_names, validation_score)))

# Evaluate your final model on the test set
test_score = distributed_model.evaluate(test_dataset)
print("Test Score:", dict(zip(distributed_model.metrics_names, test_score)))

########################################

preds = model.predict(validation_dataset)
fpr, tpr, thresholds = roc_curve(y_val, preds)
AUC = auc(fpr, tpr)
print("AUC: {:.3f}".format(AUC))

plt.subplot(1, 3, 3)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.savefig('/home/theodoropoulos/PhD/Results/ROC.png')


# # Calculate the Precision-Recall curve
# precision, recall, _ = precision_recall_curve(y_val, preds)
# pr_auc = average_precision_score(y_val, preds)
# print("precision_recall: {:.3f}".format(pr_auc))
