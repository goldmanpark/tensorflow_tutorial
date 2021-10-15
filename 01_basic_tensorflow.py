from tensorflow import keras
import numpy as np
import math


def model_sin(degree):
    x_arr = []
    y_arr = []
    for deg in range(360):
        x_arr.append(deg)
        y_arr.append(math.sin(math.pi * (deg / 180)))  # degree to radian

    xs = np.array(x_arr, dtype=int)
    ys = np.array(y_arr, dtype=float)

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=1, input_shape=[1]))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.summary()
    model.fit(xs, ys, epochs=10)
    ans = model.predict([math.pi * (degree / 180)])
    return ans[0]


prediction = model_sin(720)
realValue = math.sin(math.pi * (720 / 180))
print('prediction : ' + str(prediction))
print('real value : ' + str(realValue))
print('error : ' + str(realValue - prediction))
