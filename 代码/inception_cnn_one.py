#!usr/bin/python
import numpy as np
import os
import warnings
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
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
import matplotlib.pyplot as plt
import pylab
from matplotlib.pyplot import imshow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
K.set_image_data_format('channels_last')

X_train_orig, Y_train_orig = get_train_set()
X_test_orig, Y_test_orig = get_test_set()

index_train = np.arange(33792)
index_test = np.arange(480)
np.random.shuffle(index_train)
np.random.shuffle(index_test)
print("train:")
print(index_train)
print("test:")
print(index_test)

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

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


# inception block
def inception_block(x):
    x_1x1 = Conv2D(32, (1, 1))(x)
    x_1x1 = BatchNormalization(axis=3, epsilon=0.00001)(x_1x1)
    x_1x1 = Activation('relu')(x_1x1)

    x_3x3 = Conv2D(8, (1, 1))(x)
    x_3x3 = BatchNormalization(axis=3, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)
    x_3x3 = ZeroPadding2D(padding=(1, 1))(x_3x3)
    x_3x3 = Conv2D(16, (3, 3))(x_3x3)
    x_3x3 = BatchNormalization(axis=3, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)

    x_5x5 = Conv2D(8, (1, 1))(x)
    x_5x5 = BatchNormalization(axis=3, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)
    x_5x5 = ZeroPadding2D(padding=(2, 2))(x_5x5)
    x_5x5 = Conv2D(8, (5, 5))(x_5x5)
    x_5x5 = BatchNormalization(axis=3, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)

    x_pool = MaxPooling2D(pool_size=2, strides=2)(x)
    x_pool = Conv2D(16, (1, 1))(x_pool)
    x_pool = BatchNormalization(axis=3, epsilon=0.00001)(x_pool)
    x_pool = Activation('relu')(x_pool)
    x_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(x_pool)
    inception = keras.layers.concatenate([x_1x1, x_3x3, x_5x5, x_pool], axis=3)
    return inception


# fingerprint model
def fingerprint_model(input_shape):
    x_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(1, 1))(x_input)
    x = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    x = inception_block(x)
    x = inception_block(x)
    x = inception_block(x)

    # FC
    x = Flatten()(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=x_input, outputs=y, name='fingerprint_model')
    return model


myModel = fingerprint_model((112, 112, 1))
myModel.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                loss='binary_crossentropy', metrics=['accuracy'])

# check point
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
myModel.fit(x=X_train, y=Y_train, validation_split=0.09090909, batch_size=64, epochs=20, callbacks=callbacks_list)
preds = myModel.evaluate(x=X_test, y=Y_test)

print("TEST SET:")
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))


"""
print(np.around(happyModel.predict(X_test), decimals=1))
print(np.around(Y_test))
"""

x_test = []
img_path = 'C:\\Users\\asus\\PycharmProjects\\fh\inception\images\\1.png'
im = array(Image.open(img_path))
x_test.append(im)
img_path = 'C:\\Users\\asus\\PycharmProjects\\fh\inception\images\\2.png'
im = array(Image.open(img_path))
x_test.append(im)
img_path = 'C:\\Users\\asus\\PycharmProjects\\fh\inception\images\\019_9_4_CROP_0.png'
im = array(Image.open(img_path))
x_test.append(im)

xx_test = np.array(x_test)
x_temp = xx_test/255.
x_temp = x_temp.reshape(3, 112, 112, -1)
print("x.shape")
print(x_temp.shape)
print()
print(myModel.predict(x_temp))