from scipy.io import loadmat
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json

import cv2
import numpy as np
import keras
import string # For getting the letters of the alphabet

def trainKmodel():

	### Load EMNIST file
	mat_file_path = "./matlab/emnist-letters.mat"
	mat = loadmat(mat_file_path) # Load matlab file

	# Load char mapping
	charmap = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]} # Load character data into a file
	print(charmap)
	# pickle.dump(mapping, open('bin/mapping.p', 'wb' ))
	# max_ = None
	# Load training data
	numTrainSamples = len(mat['dataset'][0][0][0][0][0][0])

	training_images = mat['dataset'][0][0][0][0][0][0][:numTrainSamples].reshape(numTrainSamples, 28, 28, 1)
	training_labels = mat['dataset'][0][0][0][0][0][1][:numTrainSamples].reshape(numTrainSamples, )
	print("Tesla")
	print(np.unique(training_labels))

	
	# y_test = keras.utils.to_categorical(y_test, num_classes)
	
	# Load testing data
	validationSize = int(numTrainSamples / 6)
	testing_images = mat['dataset'][0][0][1][0][0][0][:validationSize].reshape(validationSize, 28, 28, 1)
	testing_labels = mat['dataset'][0][0][1][0][0][1][:validationSize].reshape(validationSize, )
	
	training_labels = keras.utils.to_categorical((training_labels - 1), len(charmap))
	testing_labels = keras.utils.to_categorical((testing_labels - 1), len(charmap))
	### KERAS MODEL


	model = Sequential()

	#add model layers
	model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28,28,1)))
	model.add(Conv2D(32, kernel_size=3, activation="relu"))
	model.add(Flatten())
	model.add(Dense(len(charmap.keys()), activation="softmax"))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(training_images, training_labels, validation_data=(testing_images, testing_labels), epochs=10, batch_size=75)
	# model.fit(training_images, training_labels, epochs=3, batch_size=75)
	# print("bock: ",training_images[0].shape)
	# print(training_labels)
	

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
	# cv2.imshow("reee",r)
	# cv2.waitKey(0)

	p = mm.predict(resiz) # Number place in the array that we think our letter is
	# print("mah prediciton:",resiz)
	print(p)
	print()
	lpred = string.ascii_lowercase[np.argmax(p[0])] # Get our letter prediction
	print("I think this is the letter: ",lpred)
	return

m = trainKmodel()

# m = testKmodel()
# print(m.summary())
# im = cv2.imread("testc.jpg")
# trypredict(mm=m,imgg=im)