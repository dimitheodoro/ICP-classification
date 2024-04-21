from Create_Patches import create_patches
from sklearn.preprocessing import LabelEncoder
import pydicom as dcm
import os
import numpy as np
import cv2
from scipy import ndimage
from segment_brain import segment,segment_all_patients_slices
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
from My_model import MultipleInputsModel
import tensorflow as tf
import random
from augmentations import CT_augmentations
from sklearn.utils.class_weight import compute_class_weight
from Accuracy_at_patients_level import accuracy_patient_level
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#os.environ["CUDA_VISIBLE_DEVICES"] ="1"
#num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
# Set environment variable to specify which GPUs to use
#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(num_gpus)])
# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()

EPOCHS = 150
patients = list(np.load("/home/theodoropoulos/PhD/Data/patients(519_120_512_512).npy",allow_pickle=True))
random.shuffle(patients)
patients = np.array(patients)

print(patients.shape)
##############################
labels_sex = np.array([patients[i]['sex']  for i in range(len(patients)) ]) #array([[0],
       #                                                                            [0]])
le_sex = LabelEncoder()
le_sex.fit(labels_sex)
labels_sex_transf = le_sex.transform(labels_sex)
labels_sex_transf #array([0, 0], dtype=int64)
# ##############################################
labels_age = np.array([patients[i]['age']  for i in range(len(patients)) ]) #array(['38', '38'], dtype='<U2')

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

def age_(age):
    if age=='NA':
        return np.array(['NA'])
    elif int(age)<30:
        return np.array(['Adult'])
    elif int(age)>=30 and int(age)<60:
        return np.array(['Middle'])
    else:
        return np.array(['Old'])

le_age = LabelEncoder()
le_age.fit(labels_age_categ)

# ####################################################################
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

def gcs_(GCS):

    if GCS=='NA':
        return np.array(['NA'])
    elif int(GCS)<=8:
        return np.array(['HIGH'])
    else:
        return np.array(['LOW'])

le_gcs = LabelEncoder()
le_gcs.fit(labels_GCS_categ)

# ########################################################
for patient in tqdm(range(len(patients))):
    volume = np.expand_dims(patients[patient]['volume'],axis=0)
    volume = np.transpose(volume,(0,2,3,1))
    volume = segment_all_patients_slices(volume)
    patients[patient]['volume']=create_patches(volume)

    patients[patient]['sex'] =  int(le_sex.transform(np.array( [patients[patient]['sex']])))
    patients[patient]['age'] =  int(le_age.transform(age_(patients[patient]['age'])))
    patients[patient]['Glasgow Coma Scale'] = int( le_gcs.transform(gcs_(patients[patient]['Glasgow Coma Scale'])))
    patients[patient]['Class']=int(patients[patient]['Class'])

X = np.squeeze(np.array([patients[i]['volume']  for i in range(len(patients)) ]),axis=1)

print(X.shape)

y = np.array (  [patients[i]['Class']  for i in range(len(patients)) ]).astype('int32')

labels_age_transf = np.array([patients[i]['age'] for i in range(len(patients))])
labels_sex_transf = np.array([patients[i]['sex'] for i in range(len(patients))])
labels_GCS_transf = np.array([patients[i]['Glasgow Coma Scale'] for i in range(len(patients))])


X_train_,  X_test_,  y_train_, y_test_,   labels_sex_train_, labels_sex_test_,  labels_age_train_, labels_age_test_ ,label_GCS_train_, label_GCS_test_= train_test_split( X, y,
                                                                                                                                                                labels_sex_transf,
                                                                                                                                                                labels_age_transf,
                                                                                                                                                                labels_GCS_transf,
                                                                                                                                                                test_size=0.2,shuffle=False)

X_train = np.reshape(X_train_,(-1,128,128,120)) ################################
X_test = np.reshape(X_test_,(-1,128,128,120))##################                     

labels_sex_train = np.repeat(labels_sex_train_, X.shape[1])
labels_sex_test =np.repeat(labels_sex_test_, X.shape[1])

labels_age_train = np.repeat(labels_age_train_, X.shape[1])
labels_age_test = np.repeat(labels_age_test_, X.shape[1])

label_GCS_train = np.repeat(label_GCS_train_, X.shape[1])
label_GCS_test = np.repeat(label_GCS_test_,X.shape[1])

label_GCS_train = np.repeat(label_GCS_train_, X.shape[1])
label_GCS_test = np.repeat(label_GCS_test_,X.shape[1])

label_GCS_train = np.repeat(label_GCS_train_, X.shape[1])
label_GCS_test = np.repeat(label_GCS_test_,X.shape[1])


y_train = np.repeat(y_train_, X.shape[1])
y_test = np.repeat(y_test_, X.shape[1])


model =MultipleInputsModel(input_shape=(128,128,120),sex_label_shape=(1,),age_label_shape=(1,),GCS_label_shape=(1,),
                           age_num_classes=len(np.unique(labels_age_transf)),
                           sex_num_classes=len(np.unique(labels_sex_transf)),
                           GCS_num_classes=len(np.unique(labels_GCS_transf)),
                           )


def train_preprocessing(volume, labels_sex,labels_age,labels_gcs,y):
# def train_preprocessing(volume,y):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    # volume = rotate(volume)
    volume = CT_augmentations(volume) #########################
    
    # volume = tf.expand_dims(volume, axis=3)
    
    return (volume, labels_sex,labels_age,labels_gcs),y
    # return volume,y

def test_preprocessing(volume, labels_sex,labels_age,labels_gcs,y):
# def test_preprocessing(volume,y):

    # volume = tf.expand_dims(volume, axis=3)
    return (volume, labels_sex,labels_age,labels_gcs),y
    # return volume,y

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((X_train,    
                                                     labels_sex_train, 
                                                     labels_age_train, 
                                                     label_GCS_train, 
                                                     y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((X_test,    
                                                     labels_sex_test, 
                                                     labels_age_test, 
                                                     label_GCS_test, 
                                                     y_test))


batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(X_train))
    .map(train_preprocessing)
    .batch(batch_size)
   .prefetch(1)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(X_test))
    .map(test_preprocessing)
    .batch(batch_size)
   .prefetch(1)
)

class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weight = {0:class_weight[0], 1:class_weight[1]}

epochs = EPOCHS
# Create and compile the distributed model
with strategy.scope():
    distributed_model = model


history = distributed_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    class_weight=class_weight,
    # callbacks=[checkpoint_cb, early_stopping_cb],
)

# Plot training and validation loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# # Plot training and validation accuracy
# plt.subplot(1, 3, 2)
# plt.plot(history.history['auc'], label='Training Accuracy')
# plt.plot(history.history['val_auc'], label='Validation Accuracy')
# plt.title('Training and Validation AUC')
# plt.xlabel('Epochs')
# plt.ylabel('AUC')
# plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Training and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

preds = distributed_model.predict(validation_dataset) ############ εδω ειχε πριν model.predict
fpr, tpr, thresholds = roc_curve(y_test, preds)
AUC = auc(fpr, tpr)
print("AUC: {:.3f}".format(AUC))

plt.subplot(1, 3, 3)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve- AUC: {:.3f}'.format(AUC))
plt.savefig('/home/theodoropoulos/PhD/Results/ROC.png')
distributed_model.save('/home/theodoropoulos/PhD/Results/patches_model.keras')

patients_test_patches =np.array([X_test[i:i+X.shape[1]] for i in range(0, len(X_test), X.shape[1])])
labels_sex_test_patches =[labels_sex_test[i:i+X.shape[1]] for i in range(0, len(labels_sex_test), X.shape[1])]
labels_age_test_patches =[labels_age_test[i:i+X.shape[1]] for i in range(0, len(labels_age_test), X.shape[1])]
labels_GCS_test_patches =[label_GCS_test[i:i+X.shape[1]] for i in range(0, len(label_GCS_test), X.shape[1])]
y_test = [y_test[i:i+X.shape[1]] for i in range(0, len(y_test), X.shape[1])]

print(y_test)
print("patient level accuracy:")
from Accuracy_at_patients_level import *
print("Accuracy: ",accuracy_patient_level(distributed_model,patients_test_patches,y_test,labels_sex_test_patches,labels_age_test_patches,labels_GCS_test_patches))
print("Recall: ",recall_patient_level(distributed_model,patients_test_patches,y_test,labels_sex_test_patches,labels_age_test_patches,labels_GCS_test_patches))
