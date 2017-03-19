from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def ModelCNN():

    nb_classes = 10

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # Model Definition
    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=(img_rows, img_cols, 1), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    return model