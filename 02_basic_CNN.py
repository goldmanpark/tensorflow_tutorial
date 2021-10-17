from tensorflow import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt

# against URL fetch failure(https://github.com/tensorflow/tensorflow/issues/33285)
import requests
import ssl
requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

class StopCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99 :
            print("Reached 99% accuracy so cancelling training!")
            self.model.stop_training = True

# create dataset
train_datagen = ImageDataGenerator(rescale=1/255)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train/255.0   #normalization
x_test=x_test/255.0
print('x_train :', np.shape(x_train))
print('y_train :', np.shape(y_train))
print('x_test :', np.shape(x_test))
print('y_test :', np.shape(y_test))

# build a CNN model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(filters=4, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, callbacks=[StopCallback()])

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))