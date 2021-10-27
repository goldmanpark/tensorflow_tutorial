import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# pre-trained word vectors
embeddings_index = {};
with open('./glove.6B/glove.6B.100d.txt', encoding='utf8') as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs

# get preprocessed data(imdb)
imdb = keras.datasets.imdb
MAX_LEN = 100
(train_sentences, train_labels), (test_sentences, test_labels) = imdb.load_data()
word_index = imdb.get_word_index()

train_padded = pad_sequences(train_sentences, maxlen=MAX_LEN, padding='post', truncating='post')
test_padded = pad_sequences(test_sentences, maxlen=MAX_LEN, padding='post', truncating='post')

EMBEDDING_DIM = MAX_LEN
VOCA_SIZE = len(word_index) + 1
embeddings_matrix = np.zeros((VOCA_SIZE, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

model = keras.Sequential()
model.add(keras.layers.Embedding(VOCA_SIZE, EMBEDDING_DIM, input_length=MAX_LEN, weights=[embeddings_matrix], trainable=False))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv1D(64, 5, activation='relu'))
model.add(keras.layers.MaxPooling1D(pool_size=4))
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_padded, train_labels, epochs=20, validation_data=(test_padded, test_labels))