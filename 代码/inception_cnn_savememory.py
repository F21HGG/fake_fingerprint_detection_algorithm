#!usr/bin/python
import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from kt_utils import *
import matplotlib.pyplot as plt
import keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings('ignore')
K.set_image_data_format('channels_last')

X_train_orig, Y_train_orig = get_train_set()
X_test_orig, Y_test_orig = get_test_set()


# inception block
def inception_block_1(x):
    x_1x1 = Conv2D(32, (1, 1))(x)
    x_1x1 = BatchNormalization(axis=3, epsilon=0.00001)(x_1x1)
    x_1x1 = Activation('relu')(x_1x1)

    x_3x3 = Conv2D(8, (1, 1))(x)
    x_3x3 = BatchNormalization(axis=3, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)
    # x_3x3 = ZeroPadding2D(padding=(1, 1))(x_3x3)
    x_3x3 = Conv2D(16, (3, 3), padding='same')(x_3x3)
    x_3x3 = BatchNormalization(axis=3, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)

    x_5x5 = Conv2D(8, (1, 1))(x)
    x_5x5 = BatchNormalization(axis=3, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)
    # x_5x5 = ZeroPadding2D(padding=(2, 2))(x_5x5)
    x_5x5 = Conv2D(8, (5, 5), padding='same')(x_5x5)
    x_5x5 = BatchNormalization(axis=3, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)

    x_pool = MaxPooling2D(pool_size=2, strides=1, padding='same')(x)
    x_pool = Conv2D(16, (1, 1))(x_pool)
    x_pool = BatchNormalization(axis=3, epsilon=0.00001)(x_pool)
    x_pool = Activation('relu')(x_pool)
    # x_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(x_pool)
    inception = keras.layers.concatenate([x_1x1, x_3x3, x_5x5, x_pool], axis=3)
    return inception


def inception_block_2(x):
    x_1x1 = Conv2D(64, (1, 1))(x)
    x_1x1 = BatchNormalization(axis=3, epsilon=0.00001)(x_1x1)
    x_1x1 = Activation('relu')(x_1x1)

    x_3x3 = Conv2D(16, (1, 1))(x)
    x_3x3 = BatchNormalization(axis=3, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)
    # x_3x3 = ZeroPadding2D(padding=(1, 1))(x_3x3)
    x_3x3 = Conv2D(32, (3, 3), padding='same')(x_3x3)
    x_3x3 = BatchNormalization(axis=3, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)

    x_5x5 = Conv2D(16, (1, 1))(x)
    x_5x5 = BatchNormalization(axis=3, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)
    # x_5x5 = ZeroPadding2D(padding=(2, 2))(x_5x5)
    x_5x5 = Conv2D(16, (5, 5), padding='same')(x_5x5)
    x_5x5 = BatchNormalization(axis=3, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)

    x_pool = MaxPooling2D(pool_size=2, strides=1)(x)
    x_pool = Conv2D(32, (1, 1))(x_pool)
    x_pool = BatchNormalization(axis=3, epsilon=0.00001)(x_pool)
    x_pool = Activation('relu')(x_pool)
    x_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(x_pool)
    inception = keras.layers.concatenate([x_1x1, x_3x3, x_5x5, x_pool], axis=3)
    return inception


def inception_block_3(x):
    x_1x1 = Conv2D(128, (1, 1))(x)
    x_1x1 = BatchNormalization(axis=3, epsilon=0.00001)(x_1x1)
    x_1x1 = Activation('relu')(x_1x1)

    x_3x3 = Conv2D(32, (1, 1))(x)
    x_3x3 = BatchNormalization(axis=3, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)
    # x_3x3 = ZeroPadding2D(padding=(1, 1))(x_3x3)
    x_3x3 = Conv2D(64, (3, 3), padding='same')(x_3x3)
    x_3x3 = BatchNormalization(axis=3, epsilon=0.00001)(x_3x3)
    x_3x3 = Activation('relu')(x_3x3)

    x_5x5 = Conv2D(32, (1, 1))(x)
    x_5x5 = BatchNormalization(axis=3, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)
    # x_5x5 = ZeroPadding2D(padding=(2, 2))(x_5x5)
    x_5x5 = Conv2D(32, (5, 5), padding='same')(x_5x5)
    x_5x5 = BatchNormalization(axis=3, epsilon=0.00001)(x_5x5)
    x_5x5 = Activation('relu')(x_5x5)

    x_pool = MaxPooling2D(pool_size=2, strides=1)(x)
    x_pool = Conv2D(64, (1, 1))(x_pool)
    x_pool = BatchNormalization(axis=3, epsilon=0.00001)(x_pool)
    x_pool = Activation('relu')(x_pool)
    x_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(x_pool)
    inception = keras.layers.concatenate([x_1x1, x_3x3, x_5x5, x_pool], axis=3)
    return inception


# fingerprint model
def fingerprint_model(input_shape):
    x_input = Input(shape=input_shape)
    # x = ZeroPadding2D(padding=(1, 1))(x_input)
    x = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    x = inception_block_1(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    x = inception_block_2(x)
    x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    x = inception_block_3(x)
    # FC
    # x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=x_input, outputs=y, name='fingerprint_model')
    return model


myModel = fingerprint_model((112, 112, 1))
myModel.compile(optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                loss='binary_crossentropy', metrics=['accuracy'])

# check point
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
history = myModel.fit(x=X_train_orig, y=Y_train_orig, validation_split=0.09090909, batch_size=64, epochs=5,
                      callbacks=callbacks_list)
plot_model(myModel, to_file='model1.png', show_shapes=True, show_layer_names=False)

myModel = load_model(filepath)
preds = myModel.evaluate(x=X_test_orig, y=Y_test_orig)
print("TEST SET:")
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

# 新建一张图
fig = plt.figure()
plt.plot(history.history['acc'], label='training acc')
plt.plot(history.history['val_acc'], label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('acc.png')
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('acc_and_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
fig.savefig('acc_and_loss.png')
fig = plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
fig.savefig('loss.png')
print("训练完毕")

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
myModel.predict_probs()
xx_test = np.array(x_test)
xx_test = xx_test.reshape(3, 112, 112, -1)
# myModel.predict_probs()
print(myModel.predict(xx_test))
"""
