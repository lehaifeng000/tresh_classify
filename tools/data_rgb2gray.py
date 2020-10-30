# 把彩色图像转为灰度图
import cv2
import os

path1 = ['../data/train/O', '../data/train/R']
path2 = ['../data/train/GRAY_O', '../data/train/GRAY_R']
# img = cv2.imread('../O_1.jpg', cv2.IMREAD_GRAYSCALE)
files = os.listdir(path1[0])
print(files)

for index, path in enumerate(path1):
    for file in os.listdir(path):
        name = path + '/' + file
        rename = path2[index] + '/' + file
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(rename, img)
        print(name+'--->'+rename)
