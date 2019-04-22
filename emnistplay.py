from scipy.io import loadmat
from keras.models import load_model


import matplotlib.pyplot as plt
import numpy as np
import string # For getting the letters of the alphabet
import keras
import cv2

### Load EMNIST file
mat_file_path = "./matlab/emnist-letters.mat"
mat = loadmat(mat_file_path) # Load matlab file

charmap = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]} # Load character data into a dict (Key: place in letter of alphabet, Value: number of occurence)
print(charmap)

## Load training and testing data
numTrainSamples = len(mat['dataset'][0][0][0][0][0][0]) #Size of training dataset


# Training data
training_images = mat['dataset'][0][0][0][0][0][0][:numTrainSamples].reshape(numTrainSamples, 28, 28, 1)
training_labels = mat['dataset'][0][0][0][0][0][1][:numTrainSamples].reshape(numTrainSamples, )


NUM = 5
print(training_images[NUM])
print(training_labels[NUM]-1)
print(string.ascii_lowercase[ training_labels[NUM]-1 ])
# lpred = string.ascii_lowercase[np.argmax(p[0])] # Get our letter prediction
cv2.imshow("reeee",training_images[NUM])
cv2.waitKey(0)

print("james prediction:")
mdl = load_model("teammodelFINAL.h5")
p = mdl.predict(np.reshape(training_images[NUM],(1,28,28,1))) # Number place in the array that we think our letter is
print(p)
print(np.argmax(p))