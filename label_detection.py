from keras.models import load_model

import numpy as np
import keras
import cv2


def predict(img, model):
	shaped_img = np.reshape(img, (1, 28, 28, 1))
	pred = model.predict(shaped_img)
	letter_num = np.argmax(pred[0])
	letter = chr(letter_num + 65)
	return letter