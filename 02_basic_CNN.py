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

# build a CNN model(VGG-style)
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(32, 32, 3)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# create dataset with data_augmentation
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.array(x_train, dtype="float") / 255.0
x_test = np.array(x_test, dtype="float") / 255.0

BATCH_SIZE = 32
datagen = ImageDataGenerator(zoom_range=0.2, rotation_range=15, horizontal_flip=True)
datagen.fit(x_train)
model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), 
                    steps_per_epoch=len(x_train) // BATCH_SIZE // 5,
                    epochs=50, 
                    callbacks=[StopCallback()], 
                    validation_data=(x_test, y_test), 
                    validation_steps=len(x_test) // BATCH_SIZE // 5)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))