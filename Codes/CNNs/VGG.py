
import matplotlib.pyplot as plt
import keras
import cv2
import numpy as np
import os
import tensorflow as tf 
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split

#Train set
imagepaths = list(paths.list_images("train"))
x_train = []
y_train = []

for imagepath in imagepaths:
  label = imagepath.split(os.path.sep)[-2]
  image = cv2.imread(imagepath)
  x_train.append(image)
  y_train.append(label)
  
x_train = np.array(x_train, 'float32')
x_train /= 255.0
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = to_categorical(y_train, 7)

#Test set
imagepaths = list(paths.list_images("test"))
x_test = []
y_test = []

for imagepath in imagepaths:
  label = imagepath.split(os.path.sep)[-2]
  image = cv2.imread(imagepath)
  x_test.append(image)
  y_test.append(label)

x_test = np.array(x_test, 'float32')
x_test /= 255.0
#le1 = LabelEncoder()
y_test = le.fit_transform(y_test)
y_test = to_categorical(y_test, 7)

import keras 
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, ZeroPadding2D, Activation, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

num_classes = 7
epochs = 20
batch_size = 256
optimizer = tf.keras.optimizers.Adam(lr=0.0005)

model = Sequential()

#1st layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,3)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

#2nd layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
model.add(BatchNormalization())

#3rd layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())

#Fully connected layer
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#model.fit_generator(x_train, y_train, epochs=epochs)
model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)

# Error
pred = model.predict(x_test, batch_size = 256)
pred = np.argmax(pred, axis=1)
pred = np.expand_dims(pred, axis=1) # make same shape as y_test
error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]

train_score = model.evaluate(x_train, y_train, verbose=0)
print('train loss:', train_score[0])
print('train accuracy:', 100*train_score[1])

test_score = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', test_score[0])
print('test accuracy:', 100*test_score[1])





