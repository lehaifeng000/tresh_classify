from tensorflow import keras
import matplotlib.pyplot as plt

# 获取手写数字数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

model = keras.Sequential()

model.add(keras.layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                              filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                              activation=keras.activations.relu))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation=keras.activations.relu))
# 分类层
model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

model.compile(optimizer=keras.optimizers.Adam(),
              # loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=1)

plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['training', 'valivation'], loc='upper left')
plt.show()

# res = model.evaluate(x_test, y_test)
res = model.predict(x_test)
print(res.shape)
print(res[0])
