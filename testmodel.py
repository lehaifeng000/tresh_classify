from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = keras.models.load_model('./model/my_model.h5')

x_train = np.load('./data/dataset/x_train.npy')
y_train = np.load('./data/dataset/y_train.npy')

x_test = np.load('./data/dataset/x_test.npy')
y_test = np.load('./data/dataset/y_test.npy')

# img=cv2.merge(x_test[0])
# cv2.imshow('jpg',x_test[0])
# cv2.waitKey(0)
# cv2.destroyWindow('test')

img = cv2.imread('./data/train/O/O_141.jpg', cv2.IMREAD_GRAYSCALE)
img_arr = np.array(img).reshape(128, 128, 1)
img_arr_float = img_arr / 255
test = np.empty((1, 128, 128, 1))
test[0] = img_arr_float

ret = model.predict(test, batch_size=1)


# ret = model.evaluate(x_train, y_train)
# ret = model.evaluate(x_test, y_test)
print(type(ret))
