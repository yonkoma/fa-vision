"""
Source of data file: https://www.nist.gov/itl/iad/image-group/emnist-dataset

EMNIST Paper: https://arxiv.org/pdf/1702.05373v1.pdf

"""

import numpy as np
import sklearn as sk

# import cPickle
import pickle as pk
import cv2

from sklearn.cluster import KMeans
import scipy.io as so

NUM_LETTERS = 26
TRAINING = False

"""
Takes EMNIST dataset and creates a model for classifiying the images that we want
"""
def trainModel():
	# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
	# print(x_train.shape)


	## Separate the data into trains sets and their associated labels
	a = so.loadmat("./matlab/emnist-letters.mat")
	# print(len(a["dataset"][0][0][0]))

	images = a["dataset"][0][0][0][0][0][0]
	labels = a["dataset"][0][0][0][0][0][1]

	print(images.shape)
	print(labels.shape)
	print(labels)
	kmeans = KMeans(n_clusters=NUM_LETTERS,n_init=1)
	kmeans.fit(images,labels)
	# exit()

	# b = np.array(images[2]).reshape(28,28)
	# print(b.shape)
	# cv2.imshow("Dice in image",b)
	# cv2.waitKey(1000)
	# cv2.destroyAllWindows()


	# save the classifier
	# with open('ocrmodel.pkl', 'wb') as fid:
	# 	cPickle.dump(kmeans, fid)
	pk.dump( kmeans, open( "ocrmodel.pkl", "wb" ) )


	return kmeans



"""
See section 2, Part C, Figure 1 of the EMNIST Paper
"""
def classifyLetter(img,kk):
	prediction = kk.predict(np.reshape(img,(-1,784)))
	print("Prediction: ",prediction)
	return

if TRAINING:
	kk = trainModel()
	exit()


# Not training, load everything and run
kk = None
kk = pk.load( open( "ocrmodel.pkl", "rb" ) )

#Test image
img = cv2.imread("b.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img)
thresh = 120
# thresholdValue,img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
thresholdValue,img = cv2.threshold(img,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

print("threshold: ",thresholdValue)
print(img.shape)
img = cv2.bitwise_not(img)

#Gaussian blur
blur = cv2.GaussianBlur(img,(5,5),0)

resized_image = cv2.resize(blur, (28, 28))
# resized_image = cv2.resize(resized_image, (-1, 784))
# print(resized_image.shape)

# cv2.imshow('ImageWindow', resized_image)
# cv2.waitKey(0)

classifyLetter(resized_image, kk)