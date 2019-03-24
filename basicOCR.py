"""
Source of data file: https://www.nist.gov/itl/iad/image-group/emnist-dataset

EMNIST Paper: https://arxiv.org/pdf/1702.05373v1.pdf

"""
import cv2
import numpy as np
import sklearn as sk

from sklearn.cluster import KMeans
import scipy.io as so

NUM_LETTERS = 26

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

	kmeans = KMeans(n_clusters=NUM_LETTERS,n_init=1)
	kmeans.fit(images,labels)
	exit()

	# b = np.array(images[2]).reshape(28,28)
	# print(b.shape)
	# cv2.imshow("Dice in image",b)
	# cv2.waitKey(1000)
	# cv2.destroyAllWindows()
	return kmeans


"""
See section 2, Part C, Figure 1 of the EMNIST Paper
"""
def classifyLetter():
	return

trainModel()
