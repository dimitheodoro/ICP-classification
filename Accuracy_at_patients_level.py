
import numpy as np

def accuracy_patient_level(model,patients_test_patches,y_test,labels_sex_test_patches,labels_age_test_patches,labels_GCS_test_patches):

    test_predictions_array = []
    for PATIENT_NUM in range(patients_test_patches.shape[0]):

        test_predictions_array.append( model.predict([patients_test_patches[PATIENT_NUM],
                                                labels_sex_test_patches[PATIENT_NUM],labels_age_test_patches[PATIENT_NUM],
                                                labels_GCS_test_patches[PATIENT_NUM]],verbose=0))
    def predict(arr):
        MEAN = np.mean(arr)
        if MEAN > 0.5:
            return 1
        else:
            return 0
    test_predictions = []

    for PATIENT_NUM in range(patients_test_patches.shape[0]):
        test_predictions.append(predict(test_predictions_array[PATIENT_NUM]))

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for label,pred in zip([y_test[i][0] for i in range(len(y_test))],test_predictions):
        print("Groundtruth:",label,"Prediction:",pred)
        if label==0 and pred==0: tn+=1
        if label==1 and pred==0: fn+=1
        if label==0 and pred==1: fp+=1
        if label==1 and pred==1: tp+=1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy

def recall_patient_level(model,patients_test_patches,y_test,labels_sex_test_patches,labels_age_test_patches,labels_GCS_test_patches):

    test_predictions_array = []
    for PATIENT_NUM in range(patients_test_patches.shape[0]):

        test_predictions_array.append( model.predict([patients_test_patches[PATIENT_NUM],
                                                labels_sex_test_patches[PATIENT_NUM],labels_age_test_patches[PATIENT_NUM],
                                                labels_GCS_test_patches[PATIENT_NUM]],verbose=0))
    def predict(arr):
        MEAN = np.mean(arr)
        if MEAN > 0.5:
            return 1
        else:
            return 0
    test_predictions = []

    for PATIENT_NUM in range(patients_test_patches.shape[0]):
        test_predictions.append(predict(test_predictions_array[PATIENT_NUM]))

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for label,pred in zip([y_test[i][0] for i in range(len(y_test))],test_predictions):
        print("Groundtruth:",label,"Prediction:",pred)
        if label==0 and pred==0: tn+=1
        if label==1 and pred==0: fn+=1
        if label==0 and pred==1: fp+=1
        if label==1 and pred==1: tp+=1

    Recall = tp/(tp+fn)
    return Recall
