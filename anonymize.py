import numpy as np

PATH = 'numpy data\patients.npy'
PATH_TO_SAVE = r'C:\Code\PAGNI_preprocessing\numpy data\anonymized_patients'

patients = np.load(PATH,allow_pickle=True)
anonymized = []
for i in range(len(patients)):
    anonymized.append(patients[i].pop('name'))

np.save(PATH_TO_SAVE,patients)
