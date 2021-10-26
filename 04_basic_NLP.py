import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# vaiables
VOCAB_SIZE = 1000
MAX_LEN = 120

# extract words, sentences from csv file
sentences = []
labels = []
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

with open("./bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ") #remove
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)

# separate train-test data
TRAIN_RATIO = 0.8
limit = int(len(sentences) * TRAIN_RATIO)
train_sentences = sentences[:limit]
train_labels = labels[:limit]
test_sentences = sentences[limit:]
test_labels = labels[limit:]

# preprocessing : tokenize(create word dictionary), integer encoding and text padding
tokenizer = Tokenizer(oov_token="<OOV>", num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(train_sentences)

train_sequence = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequence, maxlen=MAX_LEN, padding='post', truncating='post')
test_sequence = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequence, maxlen=MAX_LEN, padding='post', truncating='post')

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
train_label_sequence = np.array(label_tokenizer.texts_to_sequences(train_labels))
test_label_sequence = np.array(label_tokenizer.texts_to_sequences(test_labels))

# model
SOFTMAX_UNITS = len(label_tokenizer.word_index) + 1
model = keras.Sequential()
model.add(keras.layers.Embedding(VOCAB_SIZE, 16, input_length=MAX_LEN))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(24, activation='relu'))
model.add(keras.layers.Dense(SOFTMAX_UNITS, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_padded, train_label_sequence, epochs=20, validation_data=(test_padded, test_label_sequence))

test_loss, test_accuracy = model.evaluate(test_padded, test_label_sequence)
print("Test accuracy: {}".format(test_accuracy))