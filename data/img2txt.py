import tensorflow as tf
import numpy as np
import cv2
import os

root = os.path.abspath('./')
paths = ['./train/O/', './train/R/', './test/O/', './test/R']
img_shape = 2, 3

train_O_num = len(os.listdir(paths[0]))
train_R_num = len(os.listdir(paths[1]))
train_num = train_O_num + train_R_num
# print(train_O_num, train_R_num)

O_x_train = np.empty((train_O_num, 128, 128, 1), dtype=np.float16)
O_y_train = np.empty(train_O_num)
R_x_train = np.empty((train_R_num, 128, 128, 1), dtype=np.float16)
R_y_train = np.empty(train_R_num)

for index, file_name in enumerate(os.listdir(paths[0])):
    print(file_name)
    img = cv2.imread(paths[0] + file_name, cv2.IMREAD_GRAYSCALE)
    img_arr = np.array(img).reshape(128, 128, 1)
    img_arr_float = img_arr / 255
    O_x_train[index] = img_arr
    O_y_train[index] = 0
print(O_x_train.shape)
print(O_y_train.shape)

for index, file_name in enumerate(os.listdir(paths[1])):
    print(file_name)
    img = cv2.imread(paths[1] + file_name, cv2.IMREAD_GRAYSCALE)
    img_arr = np.array(img).reshape(128, 128, 1)
    img_arr_float = img_arr / 255
    R_x_train[index] = img_arr
    R_y_train[index] = 1
print(R_x_train.shape)
print(R_y_train.shape)

x_train = np.append(O_x_train, R_x_train, axis=0)
y_train = np.append(O_y_train, R_y_train, axis=0)
np.save('./dataset/x_train.npy', x_train)
np.save('./dataset/y_train.npy', y_train)

#
# x_train=np.shape
# for path in paths:
#     for file_name, index in enumerate(os.listdir(path)):
#         print(file_name)
#         img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
#         img_arr = np.array(img)
#         img_arr_float = img_arr / 255
#         data[index] = img_arr
