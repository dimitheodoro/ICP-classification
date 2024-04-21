from patchify import  unpatchify
import matplotlib.pyplot as plt

def reconstruct(image, patient_number=1,slice=100,TRAIN=True):
        original=image[patient_number][:,:,:,slice]
        original=original.reshape(4,4,128,128)
        return unpatchify(original,(512,512))


if __name__ == '__main__':
    icon = reconstruct(X_train_,1,78)
    plt.imshow(icon,cmap='gray')