import keras.backend as K
import math
import numpy as np
import h5py
import random
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_txt(filename, number, live_one, fake_one, fake_two, fake_three, fake_four):
    img_path = []
    img_label = []
    file = open(filename, "r")
    list = file.readlines()
    # 真指纹图像
    for i in range(live_one, live_one + number):
        ss = list[i].split()
        img_path.append(ss[0])
        img_label.append(int(ss[1]))
    # ecoflex假指纹
    for i in range(fake_one, fake_one + number // 4):
        ss = list[i].split()
        img_path.append(ss[0])
        img_label.append(int(ss[1]))
    # gelatine假指纹
    for i in range(fake_two, fake_two + number // 4):
        ss = list[i].split()
        img_path.append(ss[0])
        img_label.append(int(ss[1]))
    # latex假指纹
    for i in range(fake_three, fake_three + number // 4):
        ss = list[i].split()
        img_path.append(ss[0])
        img_label.append(int(ss[1]))
    # woodflue假指纹
    for i in range(fake_four, fake_four + number // 4):
        ss = list[i].split()
        img_path.append(ss[0])
        img_label.append(int(ss[1]))
    return img_path, img_label


# 96992  27360  27232  27680  27807
def get_train_set():
    train_img_x, train_img_y = load_txt("G:\\livdet2019\\train_set.txt", 5632, 0, 96992, 124352, 151584, 179264)
    randnum = np.random.randint(0, 100)
    np.random.seed(randnum)
    np.random.shuffle(train_img_x)
    np.random.seed(randnum)
    np.random.shuffle(train_img_y)
    for i in range(len(train_img_x)):
        train_img_x[i] = "G:\\livdet2019\\train\\"+train_img_x[i]
    train_set_x = []
    for imgpath in train_img_x:
        im = array(Image.open(imgpath))
        train_set_x.append(im)
    train_set_x_orig = np.array(train_set_x)
    train_set_x_orig = train_set_x_orig.reshape(11264, 112, 112, -1)
    train_set_y_orig = np.array(train_img_y)
    train_set_y_orig = train_set_y_orig.reshape(-1, 1)
    return train_set_x_orig, train_set_y_orig


def get_test_set():
    test_img_x, test_img_y = load_txt("G:\\livdet2019\\test_set.txt", 240, 0, 347, 407, 476, 537)
    for i in range(len(test_img_x)):
        test_img_x[i] = "G:\\livdet2019\\test\\" + test_img_x[i]
    test_set_x = []
    for imgpath in test_img_x:
        im = array(Image.open(imgpath))
        test_set_x.append(im)
    test_set_x_orig = np.array(test_set_x)
    test_set_x_orig = test_set_x_orig.reshape(480, 112, 112, -1)
    test_set_y_orig = np.array(test_img_y)
    test_set_y_orig = test_set_y_orig.reshape(-1, 1)
    return test_set_x_orig, test_set_y_orig



