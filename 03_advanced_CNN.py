from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
import os, random, shutil
from pathlib import Path

# data split
# cats/666.jpg (error)
# dogs/11702.jpg (error)
def split_data(_source, _dest_train, _dest_test):
    files = os.listdir(_source)
    limit = int(len(files) * 0.8)
    files = random.sample(files, len(files))

    for file in files[0 : limit]:
        try:
            source = _source + '/' + file
            destination = _dest_train + '/' + file
            shutil.copyfile(source, destination)
        except PermissionError:
            continue

    for file in files[limit : ]:
        try:
            source = _source + '/' + file
            destination = _dest_test + '/' + file
            shutil.copyfile(source, destination)
        except PermissionError:
            continue

SOURCE_PATH = os.getcwd() + '/kagglecatsanddogs_3367a/PetImages'
CAT_SOURCE_PATH = SOURCE_PATH + '/Cat'
DOG_SOURCE_PATH = SOURCE_PATH + '/Dog'
TRAIN_PATH =  os.getcwd() + '/kagglecatsanddogs_3367a/train'
CAT_TRAIN_PATH = TRAIN_PATH + '/Cat'
DOG_TRAIN_PATH = TRAIN_PATH + '/Dog'
TEST_PATH = os.getcwd() + '/kagglecatsanddogs_3367a/test'
CAT_TEST_PATH = TEST_PATH + '/Cat'
DOG_TEST_PATH = TEST_PATH + '/Dog'

Path(CAT_TRAIN_PATH).mkdir(parents=True, exist_ok=True)
Path(DOG_TRAIN_PATH).mkdir(parents=True, exist_ok=True)
Path(CAT_TEST_PATH).mkdir(parents=True, exist_ok=True)
Path(DOG_TEST_PATH).mkdir(parents=True, exist_ok=True)
#split_data(CAT_SOURCE_PATH, CAT_TRAIN_PATH, CAT_TEST_PATH)
#split_data(DOG_SOURCE_PATH, DOG_TRAIN_PATH, DOG_TEST_PATH)

# create data generator
datagen1 = ImageDataGenerator(rescale=1.0/255.0)
datagen2 = ImageDataGenerator(rescale=1.0/255.0)
train_gen = datagen1.flow_from_directory(directory=TRAIN_PATH, batch_size=64, class_mode='binary', target_size=(150, 150))
test_gen = datagen2.flow_from_directory(directory=TEST_PATH, batch_size=64, class_mode='binary', target_size=(150, 150))

# build a CNN model and try
model_1 = keras.models.Sequential()
model_1.add(keras.layers.InputLayer(input_shape=(150, 150, 3)))
model_1.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model_1.add(keras.layers.MaxPooling2D(2, 2))
model_1.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model_1.add(keras.layers.MaxPooling2D(2, 2))
model_1.add(keras.layers.Flatten())
model_1.add(keras.layers.Dense(128, activation='relu'))
model_1.add(keras.layers.Dropout(0.5))
model_1.add(keras.layers.Dense(1, activation='sigmoid'))
model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history1 = model_1.fit(train_gen, epochs=10, validation_data=test_gen)

# use pre-trained model(InceptionV3)
from tensorflow.keras.applications.inception_v3 import InceptionV3
pre_model = InceptionV3(input_shape=(150, 150, 3), include_top=False)
for layer in pre_model.layers:
    layer.trainable = False

temp = keras.layers.Flatten() (pre_model.output)
temp = keras.layers.Dense(128, activation='relu') (temp)
temp = keras.layers.Dropout(0.5) (temp)
temp = keras.layers.Dense(1, activation='sigmoid') (temp)
model_2 = keras.Model(inputs=pre_model.input, outputs=temp)
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = model_2.fit(train_gen, epochs=10, validation_data=test_gen)

plt.xlabel('EPOCH')
plt.ylabel('VAL_ACC')
y1 = history1.history['val_accuracy']
y2 = history2.history['val_accuracy']
plt.plot(range(len(y1)), y1, label='model1')
plt.plot(range(len(y2)), y2, label='model2')
plt.legend()
plt.show()