import os
import cv2
import numpy as np

path = './data'

os.chdir(path)
count = len(os.listdir('./'))
data = np.empty((count, 128, 128), dtype=np.float16)
for file_name, index in enumerate(os.listdir('./')):
    print(file_name)
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    img_arr = np.array(img)
    img_arr_float = img_arr / 255
    data[index] = img_arr
