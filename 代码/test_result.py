#!usr/bin/python
import os
import warnings
from keras.models import load_model
from kt_utils import *
import keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
K.set_image_data_format('channels_last')

X_test_orig, Y_test_orig = get_test_set()
live_x = X_test_orig[0:240]
live_y = Y_test_orig[0:240]
fake_x = X_test_orig[240:480]
fake_y = Y_test_orig[240:480]

filepath = "weights.best96875.hdf5"
fingerModel = load_model(filepath)
print()
preds = fingerModel.evaluate(x=X_test_orig, y=Y_test_orig)
print("All Images:"+str(X_test_orig.shape))
print("Loss = " + str(preds[0]))
print("All Images Accuracy = " + str(preds[1]))

print()
preds = fingerModel.evaluate(x=live_x, y=live_y)
print("Live Images:"+str(live_x.shape))
print("Loss = " + str(preds[0]))
print("Live Images Accuracy = " + str(preds[1]))
print("False Rejection Rate = " + str(1-preds[1]))

print()
preds = fingerModel.evaluate(x=fake_x, y=fake_y)
print("Fake Images:"+str(fake_x.shape))
print("Loss = " + str(preds[0]))
print("Fake Images Accuracy = " + str(preds[1]))
print("False Acceptance Rate = " + str(1-preds[1]))

"""
x = []
img_path = "C:\\Users\\asus\\PycharmProjects\\fh\inception\images\\1.png"
im = array(Image.open(img_path))
x.append(im)
img_path = "C:\\Users\\asus\\PycharmProjects\\fh\inception\images\\2.png"
im = array(Image.open(img_path))
x.append(im)
img_path = "C:\\Users\\asus\\PycharmProjects\\fh\inception\images\\019_9_4_CROP_0.png"
im = array(Image.open(img_path))
x.append(im)
"""

tpath = "C:\\Users\\asus\\PycharmProjects\\fh\inception\images\\"
print("请输入待测试图片")
while 1:
    imageName = input()
    if imageName == "end":
        print("结束演示")
        print()
        break
    else:
        imgTpath = tpath+imageName
        try:
            im = Image.open(imgTpath)
        except:
            print("图片地址有误，不存在该图片")
            print()
            continue
        else:
            xx = np.array(im)
            xx = xx.reshape(1, 112, 112, -1)
            num = fingerModel.predict(xx)[0][0]
            # im.show()
            print("预测值：" + str(num))
            if num >= 0.5:
                print("预测为真指纹图像")
            else:
                print("预测为假指纹图像")
            print()

"""
xx = np.array(x)
x_temp = xx.reshape(3, 112, 112, -1)
print()
print("x.shape")
print(x_temp.shape)
"""
