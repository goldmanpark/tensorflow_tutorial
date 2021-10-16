from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt


class StopCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') <= 0.005:
            print('\nReached 0.005 loss so cancelling training!')
            self.model.stop_training = True


def model_log(val):
    x_arr = np.linspace(0.1, 100, 5000)
    y_arr = np.log(x_arr)
    xs = np.array(x_arr, dtype=np.float64)
    ys = np.array(y_arr, dtype=np.float64)

    model = keras.Sequential()
    model.add(keras.layers.Dense(128, input_dim=1, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(64, activation='sigmoid'))
    model.add(keras.layers.Dense(16, activation='sigmoid'))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    model.fit(xs, ys, epochs=250, batch_size=8, callbacks=[StopCallback(), keras.callbacks.TerminateOnNaN()])
    ans = np.reshape(model.predict(val), (1, -1))
    return ans


q = np.random.rand(1, 100) * np.random.randint(100, size=100)
q = np.sort(q)
prediction = model_log(q[0])
realValue = np.log(q[0]).reshape(1, -1)
error = np.abs(realValue - prediction)
print('prediction : ' + str(prediction))
print('real value : ' + str(realValue))
print('error : ' + str(error))

plt.xlabel('x')
plt.ylabel('ln(x)')
plt.scatter(q, prediction, label='prediction')
plt.scatter(q, realValue, label='realValue')
plt.scatter(q, error, label='error')
plt.legend()
plt.show()