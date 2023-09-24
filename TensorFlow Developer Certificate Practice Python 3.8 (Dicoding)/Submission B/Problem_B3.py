# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator


def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    TRAINING_DIR = "data/rps/"
    training_datagen = ImageDataGenerator(
        rescale=1 / 255,
        validation_split=0.2
    )

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    # Load the training data into train_generator
    train_generator = training_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                           batch_size=32,
                                                           class_mode='categorical',
                                                           target_size=(150, 150),
                                                           subset='training')

    # Load the validation data into val_generator
    val_generator = training_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         target_size=(150, 150),
                                                         subset='validation')


    model=tf.keras.models.Sequential([
        # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax

        # First convolution layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3),
                               kernel_regularizer=tf.keras.regularizers.l1(0.01)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second convolution layer
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Third convolution layer
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Fourth convolution layer
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Fifth convolution layer
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten, dropout, and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Define the optimizer
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, validation_data=val_generator, epochs=10)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save("model_B3.h5")
