from __future__ import print_function

import os
from datetime import datetime
from time import time

from keras.datasets import mnist
from keras.utils import np_utils

from cnn_net import *
from utils import *


def train_loop(X_train, y_train, X_test, y_test, nb_epochs, batch_size, iteration, log_file):
    for current_epoch in range(0, nb_epochs):
        print("Number of epoch: " + str(current_epoch + 1) + "/" + str(nb_epochs))

        model.fit(X_train, y_train, batch_size=batch_size,
                  nb_epoch=1, validation_data=(X_test, y_test),
                  verbose=2)  # <-- pensar en sets de validacio augmentables

        score_train = model.evaluate(X_labeled_train, y_labeled_train, verbose=0)
        score_test = model.evaluate(X_test, y_test, verbose=0)

        log_file.write('{0} {1} {2} {3} {4} {5} {6} \n'.format(str(iteration), str(current_epoch + 1),
                                                               str(len(X_train)), str(score_train[0]),
                                                               str(score_train[1]), str(score_test[0]),
                                                               str(score_test[1])))


initial_time = time()
# Paths

model_name = "ceal_cnn_v1"
model_path = "models/" + model_name + "/"
logs_path = model_path + "/logs/"
weights_path = "models/" + model_name + "/weights/"

if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    print('Path created: ', logs_path)

if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    print('Path created: ', weights_path)

# Data Loading and Preprocessing

""" MNIST data:
    Train Set = 60,000 samples
    Validation Set = 10,000 samples """

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

##################################################

# CEAL params

nb_labeled = 600
nb_unlabeled = X_train.shape[0] - nb_labeled

nb_iterations = 1
nb_annotations = 200
initial_decay_rate = 0.6
decay_rate = 0.5
thresh = None

nb_initial_epochs = 1
nb_active_epochs = 1
batch_size = 128
nb_classes = 10

##################################################

# DB definition

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

X_labeled_train = X_train[0:nb_labeled]
y_labeled_train = y_train[0:nb_labeled]
X_unlabeled_train = X_train[nb_labeled:len(X_train)]

# (1) Initialize model
iteration = 0

model = ModelCNN()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

data = datetime.now()
f = open(logs_path + model_name + "_log" + str(data.month) + str(data.day) + str(data.hour) + str(data.minute),
         'a')
f.write("Log format: iteration / epoch / nb_train / loss_train / acc_train / loss_test / acc_test\n"
        "iteration: 0 - Initialization / ~ 0 - Active training\n\n")

train_loop(X_labeled_train, y_labeled_train, X_test, y_test, nb_initial_epochs, batch_size, iteration, f)


# Active loop

for iteration in range(1, nb_iterations + 1):
    # (2) Labeling

    print("Getting predictions...")
    t = time()
    predictions = model.predict(X_unlabeled_train, verbose=0)
    print("Time elapsed: " + str(time() - t) + " s")

    predictions_rank, en = entropy_rank(predictions)

    # labeling by Oracle process: MNIST case
    uncertain_samples = uncertain_set(predictions_rank, nb_annotations)
    y_oracle_train = y_train[uncertain_samples]
    X_oracle_train = X_unlabeled_train[uncertain_samples]
    np.delete(X_unlabeled_train, uncertain_samples)
    np.delete(en, uncertain_samples)

    # pseudo-labeling
    certain_samples, thresh = certain_set(en, thresh, initial_decay_rate, decay_rate)

    #certain_samples, thresh = certain_set(en, thresh, 4)

    print("Thresh = " + str(thresh))
    print("Certain samples = " + str(len(certain_samples)))

    X_pseudo_train = X_unlabeled_train[certain_samples]
    y_pseudo_train = predictions_max_class(predictions[certain_samples], nb_classes)

    print("Pseudo_labeling error = "+str(pseudo_label_error(y_pseudo_train,y_train[nb_labeled+certain_samples])))

    X_labeled_train_aux = np.concatenate((X_labeled_train, X_oracle_train, X_pseudo_train))
    y_labeled_train_aux = np.concatenate((y_labeled_train, y_oracle_train, y_pseudo_train))

    # (3) Training
    train_loop(X_labeled_train_aux, y_labeled_train_aux, X_test, y_test, nb_active_epochs, batch_size, iteration, f)
    X_labeled_train = np.concatenate((X_labeled_train, X_oracle_train))
    y_labeled_train = np.concatenate((y_labeled_train, y_oracle_train))

print("Time elapsed: " + str(time() - initial_time) + " s")
f.close()
