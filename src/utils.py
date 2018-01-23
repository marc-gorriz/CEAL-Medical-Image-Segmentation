from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.morphology import distance_transform_edt as edt

from constants import *
from unet import get_unet


def range_transform(sample):
    if (np.max(sample) == 1):
        sample = sample * 255

    m = 255 / (np.max(sample) - np.min(sample))
    n = 255 - m * np.max(sample)
    return (m * sample + n) / 255


def predict(data, model):
    return model.predict(data, verbose=0)


def compute_uncertain(sample, prediction, model):
    X = np.zeros([1, img_rows, img_cols])

    for t in range(nb_step_predictions):
        prediction = model.predict(sample, verbose=0).reshape([1, img_rows, img_cols])
        X = np.concatenate((X, prediction))

    X = np.delete(X, [0], 0)

    if (apply_edt):
        var = np.var(X, axis=0)
        transform = range_transform(edt(prediction))
        return np.sum(var * transform)

    else:
        return np.sum(np.var(X, axis=0))


def interval(data, start, end):
    p = np.where(data >= start)[0]
    return p[np.where(data[p] < end)[0]]


def get_pseudo_index(uncertain, nb):
    h = np.histogram(uncertain, 80)

    pseudo = interval(uncertain, h[1][np.argmax(h[0])], h[1][np.argmax(h[0]) + 1])
    np.random.shuffle(pseudo)
    return pseudo[0:nb]


def random_index(uncertain, nb_random):

    histo = np.histogram(uncertain, 80)
    # TODO: automatic selection of random range
    index = interval(uncertain, histo[1][np.argmax(histo[0]) + 6], histo[1][len(histo[0]) - 33])
    np.random.shuffle(index)
    return index[0:nb_random]


def no_detections_index(rank, nb_no_detections):
    return rank[0:nb_no_detections]


def most_uncertain_index(uncertain, nb_most_uncertain, rate):
    data = np.array([]).astype('int')

    histo = np.histogram(uncertain, 80)

    p = np.arange(len(histo[0]) - rate, len(histo[0]))  # index of last bins above the rate
    pr = np.argsort(histo[0][p])  # p index accendent sorted
    cnt = 0
    pos = 0
    index = np.array([]).astype('int')

    while (cnt < nb_most_uncertain and pos < len(pr)):
        sbin = histo[0][p[pr[pos]]]

        index = np.append(index, p[pr[pos]])
        cnt = cnt + sbin
        pos = pos + 1

    for i in range(0, pos):
        data = np.concatenate((data, interval(uncertain, histo[1][index[i]], histo[1][index[i] + 1])))

    np.random.shuffle(data)
    return data[0:nb_most_uncertain]



def get_oracle_index(uncertain, rank, nb_no_detections, nb_random, nb_most_uncertain, rate):
    return np.concatenate((no_detections_index(rank, nb_no_detections), random_index(uncertain, nb_random),
                           most_uncertain_index(uncertain, nb_most_uncertain, rate)))


def compute_dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.reshape([1, img_rows * img_cols])
    y_pred_f = y_pred.reshape([1, img_rows * img_cols])
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def compute_train_sets(X_train, y_train, labeled_index, unlabeled_index, weights, iteration):
    print("\nActive iteration " + str(iteration))
    print("-" * 50 + "\n")

    # load models
    modelUncertain = get_unet(dropout=True)
    modelUncertain.load_weights(weights)
    modelPredictions = get_unet(dropout=False)
    modelPredictions.load_weights(weights)

    # predictions
    print("Computing log predictions ...\n")
    predictions = predict(X_train[unlabeled_index], modelPredictions)

    uncertain = np.zeros(len(unlabeled_index))
    accuracy = np.zeros(len(unlabeled_index))

    print("Computing train sets ...")
    for index in range(0, len(unlabeled_index)):

        if index % 100 == 0:
            print("completed: " + str(index) + "/" + str(len(unlabeled_index)))

        sample = X_train[unlabeled_index[index]].reshape([1, 1, img_rows, img_cols])

        sample_prediction = cv2.threshold(predictions[index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')

        accuracy[index] = compute_dice_coef(y_train[unlabeled_index[index]][0], sample_prediction)
        uncertain[index] = compute_uncertain(sample, sample_prediction, modelUncertain)

    rank = np.argsort(uncertain)

    np.save(global_path + "logs/uncertain" + str(iteration), uncertain)
    np.save(global_path + "logs/accuracy" + str(iteration), accuracy)

    oracle_index = get_oracle_index(uncertain, rank, nb_no_detections, nb_random, nb_most_uncertain, rate, iteration)
    oracle_rank = unlabeled_index[oracle_index]

    np.save(global_path + "ranks/oracle" + str(iteration), oracle_rank)
    np.save(global_path + "ranks/oraclelogs" + str(iteration), oracle_index)

    labeled_index = np.concatenate((labeled_index, oracle_rank))

    if (iteration >= pseudo_epoch):

        pseudo_index = get_pseudo_index(uncertain, nb_pseudo_initial + (pseudo_rate * (iteration - pseudo_epoch)))
        pseudo_rank = unlabeled_index[pseudo_index]

        np.save(global_path + "ranks/pseudo" + str(iteration), pseudo_rank)
        np.save(global_path + "ranks/pseudologs" + str(iteration), pseudo_index)

        X_labeled_train = np.concatenate((X_train[labeled_index], X_train[pseudo_index]))
        y_labeled_train = np.concatenate((y_train[labeled_index], predictions[pseudo_index]))

    else:
        X_labeled_train = np.concatenate((X_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])
        y_labeled_train = np.concatenate((y_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])

    unlabeled_index = np.delete(unlabeled_index, oracle_index, 0)

    return X_labeled_train, y_labeled_train, labeled_index, unlabeled_index


def data_generator():
    return ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.2,
        rotation_range=40,
        horizontal_flip=True)


def log(history, step, log_file):
    for i in range(0, len(history.history["loss"])):
        if len(history.history.keys()) == 4:
            log_file.write('{0} {1} {2} {3} \n'.format(str(step), str(i), str(history.history["loss"][i]),
                                                       str(history.history["val_dice_coef"][i])))

def create_paths():
    path_ranks = global_path + "ranks/"
    path_logs = global_path + "logs/"
    path_plots = global_path + "plots/"
    path_models = global_path + "models/"

    if not os.path.exists(path_ranks):
        os.makedirs(path_ranks)
        print("Path created: ", path_ranks)

    if not os.path.exists(path_logs):
        os.makedirs(path_logs)
        print("Path created: ", path_logs)

    if not os.path.exists(path_plots):
        os.makedirs(path_plots)
        print("Path created: ", path_plots)

    if not os.path.exists(path_models):
        os.makedirs(path_models)
        print("Path created: ", path_models)