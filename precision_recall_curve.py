from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
from segment_brain import segment,segment_all_patients_slices
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from My_model2 import MultipleInputsModel_TURBO
from augmentations import CT_augmentations
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import auc

model= keras.models.load_model('/home/theodoropoulos/PhD/Results/final_model.keras')

##########################
patients = np.load("/home/theodoropoulos/PhD/Data/patients(578_120_128_128).npy",allow_pickle=True)

#desired_volume_dims_after_resampling = (120,256,256)

X = np.array([patients[i]['volume']  for i in range(len(patients)) ])
X =np.transpose(X,(0,2,3,1))
X = segment_all_patients_slices(X)
print(X.shape)
y = np.array (  [patients[i]['Class']  for i in range(len(patients)) ]).astype('int32')

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
##############################
model =MultipleInputsModel_TURBO(input_shape=(128,128,120),sex_label_shape=(1,),age_label_shape=(1,),GCS_label_shape=(1,),
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
    volume = CT_augmentations(volume)
    
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


batch_size = 2

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
    validation_loader.shuffle(len(X_test))
    .map(test_preprocessing)
    .batch(batch_size)
    .prefetch(1)
)

########################################
preds = model.predict(validation_dataset)
fpr, tpr, thresholds = roc_curve(y_val, preds)
AUC = auc(fpr, tpr)
print("AUC: {:.3f}".format(AUC))

# Calculate the Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_val, preds)
pr_auc = average_precision_score(y_val, preds)
print("precision_recall: {:.3f}".format(pr_auc))

# Plot the Precision-Recall curve
plt.subplot(1,2,2)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.savefig('/home/theodoropoulos/PhD/Results/precision_recall_curve.png')

# print(preds,y_val)


# fpr, tpr, thresholds = roc_curve(y_val, preds,pos_label=0)
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'y--')
# plt.plot(fpr, tpr, marker='.')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.savefig('/home/theodoropoulos/Desktop/Results/ROC.png')
