from informative import entropyrank
from tflearn.models import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

import numpy as np
import tflearn.datasets.mnist as mnist

def network():

    layer_input = input_data(shape=[None, 28, 28, 1], name='input')

    layer_conv1 = conv_2d(layer_input, 32, 5, activation='relu')
    layer_pool1 = max_pool_2d(layer_conv1, 2)

    layer_conv2 = conv_2d(layer_pool1, 64, 5, activation='relu')
    layer_pool2 = max_pool_2d(layer_conv2, 2)

    layer_fc1 = fully_connected(layer_pool2, 1024, activation='relu')
    layer_fc1 = dropout(layer_fc1, 0.8)

    layer_fc2 = fully_connected(layer_fc1, 10, activation='softmax')
    output = regression(layer_fc2, optimizer='sgd', learning_rate=0.01,
                        loss='categorical_crossentropy', name='targets')

    model = DNN(output)

    return model

def train(model, X, Y, testX, testY):

    model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': testX}, {'targets': testY}),
               show_metric=True, snapshot_epoch=False)

    return model

def plot_image(image):
    plt.imshow(image.reshape(28,28),cmap='binary') #mnist sizes, to change!
    plt.show()