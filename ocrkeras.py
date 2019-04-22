from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
from scipy.io import loadmat

import matplotlib.pyplot as plt
import numpy as np
import string # For getting the letters of the alphabet
import keras
import cv2

BATCH_SIZE = 4000

def trainKmodel():

	### Load EMNIST file
	mat_file_path = "./matlab/emnist-letters.mat"
	mat = loadmat(mat_file_path) # Load matlab file
	
	charmap = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]} # Load character data into a dict (Key: place in letter of alphabet, Value: number of occurence)
	print(charmap)
	
	## Load training and testing data
	numTrainSamples = len(mat['dataset'][0][0][0][0][0][0]) #Size of training dataset
	validationSize = int(numTrainSamples / 6) #Size of validation dataset

	# Training data
	training_images = mat['dataset'][0][0][0][0][0][0][:numTrainSamples].reshape(numTrainSamples, 28, 28, 1)
	training_labels = mat['dataset'][0][0][0][0][0][1][:numTrainSamples].reshape(numTrainSamples, )
	
	# Testing data
	testing_images = mat['dataset'][0][0][1][0][0][0][:validationSize].reshape(validationSize, 28, 28, 1)
	testing_labels = mat['dataset'][0][0][1][0][0][1][:validationSize].reshape(validationSize, )
	
	# Represent the data as categorical data (labels)
	training_labels = keras.utils.to_categorical((training_labels - 1), len(charmap)) # Should have shape of (numTrainSamples)
	testing_labels = keras.utils.to_categorical((testing_labels - 1), len(charmap))

	### KERAS MODEL

	# Create classification model
	model = Sequential()

	# Layers for the model
	'''
	model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28,28,1)))
	model.add(Conv2D(32, kernel_size=3, activation="relu"))
	model.add(Flatten())
	model.add(Dense(len(charmap.keys()), activation="softmax"))
	'''
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(len(charmap.keys()), activation='softmax'))

	# Specify how the model trains
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Train the model
	history = model.fit(training_images, training_labels, validation_data=(testing_images, testing_labels), epochs=10, batch_size=BATCH_SIZE)
	# model.fit(training_images, training_labels, epochs=3, batch_size=75)
	
	#Save the model so we do not need to retrain it all later
	model.save("teammodel.h5")


	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()
	
	# Return the model in case we want to use it now
	return model

def testKmodel():
	model = load_model("teammodelFINAL.h5")
	
	print("OCR Model Loaded")
	print(model.summary())

	return model


"""

"""
def trypredict(mdl,imgg):
	img = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

	# roi = cv2.selectROI(img)

	# (col, row, dcol, drow) = roi
	# cropped = image[row:row+drow,col:col+dcol]

	thresh = 120
	# thresholdValue,img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
	thresholdValue,img = cv2.threshold(img,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	print("threshold: ",thresholdValue)
	print(img.shape)
	img = cv2.bitwise_not(img)
	
	#Gaussian blur
	blur = cv2.GaussianBlur(img,(5,5),0)
	r = cv2.resize(blur, (28, 28))
	print(r.shape)
	resiz = np.reshape(r, (1,28,28,1))
	cv2.imshow("reee",r)
	cv2.waitKey(0)

	p = mdl.predict(resiz) # Number place in the array that we think our letter is
	# print("mah prediciton:",resiz)
	print(p)
	print()

	lpred = string.ascii_lowercase[np.argmax(p[0])] # Get our letter prediction
	print("I think this is the letter: ",lpred)

	return

# m = trainKmodel()

m = testKmodel()
# print(m.summary())
im = cv2.imread("testa.jpg")
trypredict(mdl=m,imgg=im)