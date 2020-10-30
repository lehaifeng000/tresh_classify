import cv2
import numpy as np
from PIL import Image


# 使用opencv读取图片(灰度图),返回numpy数组
def readImage(filePath):
    img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
    array = np.array(img)
    # print('大小：{}'.format(img.shape))
    # print("类型：%s" % type(img))
    # print(img)
    return array

# # 显示图片
# def printImage(img):
#     cv2.imshow(img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
