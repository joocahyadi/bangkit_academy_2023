# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"

    # Split the data into the training and testing set
    training_sentences, testing_sentences = train_test_split(bbc['text'], test_size=(1-training_portion), shuffle=False)
    training_labels, testing_labels = train_test_split(bbc['category'], test_size=(1-training_portion), shuffle=False)

    # Define the tokenizer for sentences and categories
    sentences_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    sentences_tokenizer.fit_on_texts(training_sentences)

    categories_tokenizer = Tokenizer()
    categories_tokenizer.fit_on_texts(bbc['category'])

    # Apply the tokenizer
    training_sentences = sentences_tokenizer.texts_to_sequences(training_sentences)
    testing_sentences = sentences_tokenizer.texts_to_sequences(testing_sentences)

    training_labels = np.array(categories_tokenizer.texts_to_sequences(training_labels))
    testing_labels = np.array(categories_tokenizer.texts_to_sequences(testing_labels))

    # Pad the sequences (for sentences only)
    training_sentences = pad_sequences(training_sentences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
    testing_sentences = pad_sequences(testing_sentences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

    model = tf.keras.Sequential([
        # YOUR CODE HERE.

        # Embedding layer
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),

        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    # Train the model
    model.fit(x=training_sentences, y=training_labels, validation_data=(testing_sentences, testing_labels), epochs=50)

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    # model.save("model_B4.h5")

