import cv2
import pickle
import os.path
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
import os
import imutils


# HDF5 supports n-dimensional datasets and each element in the dataset may itself be a complex object.
MODEL_FILENAME = "captcha_model.hdf5"

# Data file of the model labels
MODEL_LABELS_FILENAME = "model_labels.dat"

def resize_to_fit(image, width, height):

    (h, w) = image.shape[:2]

    # Resize the image according to which parameter is bigger
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)

    # Simple functions to add required padding
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # Bordering the image with the padding
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    return image

# The destination of the train dataset
folder = "extracted_letter_images"

data = []
labels = []

# Go to individual folder
for foldername in os.listdir(folder):
	path = folder+"/"+foldername
	# In that go to the individual file
	print path
	for filename in os.listdir(path):
		img = cv2.imread(path+"/"+filename,0)
		img = resize_to_fit(img, 20, 20)
		# Adds a new dimension to make it a 3D object
		img = np.expand_dims(img, axis=2)
		# Put the label of the data as the folder in which the image file is present
		label = foldername
		data.append(img)
		labels.append(label)

# Scale the pixel values between [0,1], helps in training
data = np.array(data)/255.0
labels = np.array(labels)

# 25% goes to the validation set
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Labelise into binary data, apply one-hot encoding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# A data file
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Applied CNN

model = Sequential()

model.add(Conv2D(20, (5, 5), padding="same", activation="relu",input_shape=(20,20,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
# A single hidden layer with 500 nodes
model.add(Dense(500, activation="relu"))

# 32 Output Categories
model.add(Dense(32, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

model.save(MODEL_FILENAME)



		



