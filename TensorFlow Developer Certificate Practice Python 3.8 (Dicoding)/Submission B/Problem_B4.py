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

    # Split the bbc['text'] into training and validation sentences
    training_sentences, validation_sentences = train_test_split(bbc['text'], test_size=(1 - training_portion),
                                                                shuffle=False)

    # Fit your tokenizer with training data
    sentences_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    sentences_tokenizer.fit_on_texts(training_sentences)

    training_sentences = sentences_tokenizer.texts_to_sequences(training_sentences)
    validation_sentences = sentences_tokenizer.texts_to_sequences(validation_sentences)

    # Pad the sequences
    training_sentences = pad_sequences(training_sentences, maxlen=max_length, truncating=trunc_type,
                                       padding=padding_type)
    validation_sentences = pad_sequences(validation_sentences, maxlen=max_length, truncating=trunc_type,
                                         padding=padding_type)
    
    # You can also use Tokenizer to encode your label.
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(bbc['category'])

    # Split the labels into training_labels and validation_labels
    training_labels, validation_labels = train_test_split(bbc['category'], test_size=(1 - training_portion),
                                                          shuffle=False)
    # Convert the labels into sequences
    training_labels = np.array(label_tokenizer.texts_to_sequences(training_labels))
    validation_labels = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    # Train the model
    model.fit(x=training_sentences, y=training_labels, validation_data=(validation_sentences, validation_labels), epochs=50)

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")

