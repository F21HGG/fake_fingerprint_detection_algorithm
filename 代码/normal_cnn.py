#!usr/bin/python
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
warnings.filterwarnings('ignore')

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import regularizers
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
import keras.backend as K
import keras
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
import pylab
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig = get_train_set()
X_test_orig, Y_test_orig = get_test_set()

index_train = np.arange(33792)
index_test = np.arange(480)
np.random.shuffle(index_train)
np.random.shuffle(index_test)

X_train_orig = X_train_orig[index_train, :, :, :]
Y_train_orig = Y_train_orig[index_train]
X_test_orig = X_test_orig[index_test, :, :, :]
Y_test_orig = Y_test_orig[index_test]

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig
Y_test = Y_test_orig


# GRADED FUNCTION: HappyModel
def HappyModel(input_shape):
    X_input = Input(shape=input_shape)
    X = ZeroPadding2D(padding=(1, 1))(X_input)  #padding
    X = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X)        # conv
    X = BatchNormalization(axis=3)(X)           #BN
    #X = Dropout(0.4)(X)
    X = Activation('selu')(X)    #relu
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    #X = Dropout(0.4)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    #X = Dropout(0.4)(X)
    X = Activation('selu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    # FC
    X = Flatten()(X)
    Y = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=X_input, outputs=Y, name='HappyModel')
    return model


happyModel = HappyModel((112, 112, 1))
happyModel.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])

# check point
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
happyModel.fit(x=X_train, y=Y_train, validation_split=0.09090909, batch_size=64, epochs=20, callbacks=callbacks_list)
happyModel = load_model(filepath)
preds = happyModel.evaluate(x=X_test, y=Y_test)

print("TEST SET:")
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

"""
print(np.around(happyModel.predict(X_test), decimals=1))
print(np.around(Y_test))
"""
x = []
img_path = 'C:\\Users\\asus\\PycharmProjects\\fh\inception\images\\1.png'
im = array(Image.open(img_path))
x.append(im)
img_path = 'C:\\Users\\asus\\PycharmProjects\\fh\inception\images\\2.png'
im = array(Image.open(img_path))
x.append(im)
img_path = 'C:\\Users\\asus\\PycharmProjects\\fh\inception\images\\019_9_4_CROP_0.png'
im = array(Image.open(img_path))
x.append(im)

xx = np.array(x)
x_temp = xx/255.
x_temp = x_temp.reshape(3, 112, 112, -1)
print("x.shape")
print(x_temp.shape)
print()
print(happyModel.predict(x_temp))

