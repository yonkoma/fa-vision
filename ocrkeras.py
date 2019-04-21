from scipy.io import loadmat
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json

import numpy as np
import keras
import string # For getting the letters of the alphabet

def trainKmodel():
	mat_file_path = "./matlab/emnist-letters.mat"
	mat = loadmat(mat_file_path)

	# Load char mapping
	charmap = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
	print(charmap)
	# pickle.dump(mapping, open('bin/mapping.p', 'wb' ))
	max_ = None
	# Load training data
	numTrainSamples = len(mat['dataset'][0][0][0][0][0][0])
	max_ = len(mat['dataset'][0][0][0][0][0][0])

	training_images = mat['dataset'][0][0][0][0][0][0][:numTrainSamples].reshape(numTrainSamples, 28, 28, 1)
	training_labels = mat['dataset'][0][0][0][0][0][1][:numTrainSamples].reshape(numTrainSamples, )
	print("Tesla")
	print(np.unique(training_labels))

	training_labels = keras.utils.to_categorical((training_labels - 1), len(charmap))
	# y_test = keras.utils.to_categorical(y_test, num_classes)
	'''
	# Load testing data
	if max_ == None:
	    max_ = len(mat['dataset'][0][0][1][0][0][0])
	else:
	    max_ = int(max_ / 6)
	testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, 28, 28, 1)
	testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]
	'''

	### KERAS MODEL


	model = Sequential()

	#add model layers
	model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28,28,1)))
	model.add(Conv2D(32, kernel_size=3, activation="relu"))
	model.add(Flatten())
	model.add(Dense(len(charmap.keys()), activation="softmax"))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# model.fit(training_images, training_labels, validation_data=(X_test, y_test), epochs=3)
	print("bock: ",training_images[0].shape)
	print(training_labels)
	model.fit(training_images, training_labels, epochs=3, batch_size=75)

	model.save("teammodel.h5")

	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("ocrkerasmodel.h5")

def testKmodel():
	# loaded_model = model_from_json("model.json")
	# loaded_model.load_weights("ocrkerasmodel.h5")
	# print(loaded_model)
	model = load_model("teammodel.h5")
	print(model)
	return model

def trypredict(mm,imgg):
	img = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

	roi = cv2.selectROI(img)

	(col, row, dcol, drow) = roi
	cropped = image[row:row+drow,col:col+dcol]

	thresh = 120
	# thresholdValue,img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
	thresholdValue,img = cv2.threshold(img,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	print("threshold: ",thresholdValue)
	print(img.shape)
	img = cv2.bitwise_not(img)
	#Gaussian blur
	blur = cv2.GaussianBlur(img,(5,5),0)
	resiz = cv2.resize(blur, (28, 28))
	p = mm.predict(resiz) # Number place in the array that we think our letter is
	print("mah prediciton:",resiz)
	lpred = string.ascii_lowercase[p] # Get our letter prediction
	print("I think this is the letter: ",lpred)
	return

# m = trainKmodel()

m = testKmodel()
