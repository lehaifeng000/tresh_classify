import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
root = os.path.abspath('./')
paths = ['./test/O/', './test/R/']

test_O_num = len(os.listdir(paths[0]))
test_R_num = len(os.listdir(paths[1]))
test_num = test_O_num + test_R_num
# print(test_O_num, test_R_num)

O_x_test = np.empty((test_O_num, 128, 128, 1), dtype=np.float16)
O_y_test = np.empty(test_O_num)
R_x_test = np.empty((test_R_num, 128, 128, 1), dtype=np.float16)
R_y_test = np.empty(test_R_num)

for index, file_name in enumerate(os.listdir(paths[0])):
    print(file_name)
    img = cv2.imread(paths[0] + file_name, cv2.IMREAD_GRAYSCALE)
    img_arr = np.array(img).reshape(128, 128, 1)
    img_arr_float = img_arr / 255
    O_x_test[index] = img_arr
    O_y_test[index] = 0
print(O_x_test.shape)
print(O_y_test.shape)


for index, file_name in enumerate(os.listdir(paths[1])):
    print(file_name)
    img = cv2.imread(paths[1] + file_name, cv2.IMREAD_GRAYSCALE)

    img_arr = np.array(img).reshape(128, 128, 1)
    img_arr_float = img_arr / 255
    R_x_test[index] = img_arr
    R_y_test[index] = 1
print(R_x_test.shape)
print(R_y_test.shape)

x_test = np.append(O_x_test, R_x_test, axis=0)
y_test = np.append(O_y_test, R_y_test, axis=0)
np.save('./dataset/x_test.npy', x_test)
np.save('./dataset/y_test.npy', y_test)


# x_test=np.shape
# for path in paths:
#     for file_name, index in enumerate(os.listdir(path)):
#         print(file_name)
#         img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
#         img_arr = np.array(img)
#         img_arr_float = img_arr / 255
#         data[index] = img_arr
