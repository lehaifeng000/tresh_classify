import tensorflow as tf
import numpy as np
import cv2
from collections import Counter

# data = np.ones((2, 2, 2))

# img1 = np.array(cv2.imread('./data/O_1.jpg', cv2.IMREAD_GRAYSCALE))
#
# np.save('./data.npy', data)
# a = np.loadtxt('./a.txt')
# print(a.shape)
# print(data.shape)
data=np.load('./data.npy')
print(data)
