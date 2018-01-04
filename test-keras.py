from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('tf')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#Load already shuffled MNIST data sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# Convert to float32 and normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Select model
model = Sequential()
# model.summary()

# add first layer of model and verify
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.summary()

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Add fully connected dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# Fit model
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

# Score model
score = model.evaluate(X_test, Y_test, verbose=0)
