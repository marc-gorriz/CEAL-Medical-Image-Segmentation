from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt

from constants import *
from unet import get_unet


def uncertain_rank(data, nb_step_predictions, uncertain_method, weights):

    model = get_unet(dropout=True)
    model.load_weights(weights)

    rank = np.zeros(len(data))

    for sample in range(0, len(data)):
        X = np.zeros([1, img_rows, img_cols])
        for time in range(nb_step_predictions):

            prediction = model.predict(data[sample].reshape([1, 1, img_rows, img_cols]), verbose=0).reshape(
                [1, img_rows, img_cols])

            if uncertain_method == "entropy":
                prediction[0][np.where(prediction[0] < 0.5)] = 0
                prediction[0][np.where(prediction[0] >= 0.5)] = 1

            X = np.concatenate((X, prediction))

        X = np.delete(X, [0], 0)

        if uncertain_method == "variance":
            if (apply_edt):
                var = np.var(X, axis=0)
                rank[sample] = np.sum(((255-edt(var))/255)*var)
            else:
                rank[sample] = np.sum(np.var(X, axis=0))

        if uncertain_method == "entropy":
            fg = np.count_nonzero(X, axis=0) / nb_step_predictions
            bg = 1 - fg

            pos = np.ones([X[0].shape[0], X[0].shape[1]])
            pos[np.where(fg == 0)] = 0
            pos[np.where(fg == 1)] = 0

            en = np.zeros([X[0].shape[0], X[0].shape[1]])
            en[pos != 0] = -(fg[pos != 0] * np.log(fg[pos != 0]) + bg[pos != 0] * np.log(bg[pos != 0]))

            if (apply_edt):
                rank[sample] = np.sum(((255-edt(en))/255)*en)
            else:
                rank[sample] = np.sum(en)

        if sample % 100 == 0:
            print("Done: " + str(sample))

    sorted_rank = np.argsort(rank)[::-1]

    return sorted_rank


def predict(data, weights):
    model = get_unet(dropout=False)
    model.load_weights(weights)

    return model.predict(data, verbose=0)


def compute_log_uncertain(sample, nb_step_predictions, uncertain_method, weights):
    X = np.zeros([1, img_rows, img_cols])
    model = get_unet(dropout=True)
    model.load_weights(weights)

    if uncertain_method == "variance":
        for t in range(nb_step_predictions):
            prediction = model.predict(sample.reshape([1, 1, img_rows, img_cols]), verbose=0).reshape(
                [1, img_rows, img_cols])
            X = np.concatenate((X, prediction))

    X = np.delete(X, [0], 0)
    return np.var(X, axis=0).reshape([1, img_rows, img_cols])


def save_logs(predictions_rank, X_unlabeled_train, active_step, weights):

    step_path = path_log+str(active_step)+"/"
    if not os.path.exists(step_path):
        os.makedirs(step_path)
        print('Path created: ', step_path)

    # oracle index & pseudo index
    oracle_index = predictions_rank[0:nb_annotations]

    if (sel_random):
        delta = np.arange(nb_pseudo)
        np.random.shuffle(delta)
        pseudo_index = predictions_rank[len(predictions_rank) - nb_pseudo:len(predictions_rank)]
        pseudo_index = pseudo_index[delta[0:nb_log_pseudo]]

    else:
        pseudo_index = predictions_rank[len(predictions_rank) - nb_log_pseudo:len(predictions_rank)]

    np.save(step_path + "oracle_index.npy", oracle_index)
    np.save(step_path + "pseudo_index.npy", pseudo_index)

    # predictions (pseudo_index)
    np.save(step_path + "oracle_predictions.npy", predict(X_unlabeled_train[oracle_index], weights))
    np.save(step_path + "pseudo_predictions.npy", predict(X_unlabeled_train[pseudo_index], weights))

    # uncertain (oracle_index & pseudo_index)
    oracle_uncertain = np.zeros([1, img_rows, img_cols])
    for sample in X_unlabeled_train[oracle_index]:
        oracle_uncertain = np.concatenate(
            (oracle_uncertain, compute_log_uncertain(sample, nb_step_predictions, uncertain_method, weights)))

    np.save(step_path + "oracle_uncertain.npy", np.delete(oracle_uncertain, [0], 0))

    pseudo_uncertain = np.zeros([1, img_rows, img_cols])
    for sample in X_unlabeled_train[pseudo_index]:
        pseudo_uncertain = np.concatenate(
            (pseudo_uncertain, compute_log_uncertain(sample, nb_step_predictions, uncertain_method, weights)))

    np.save(step_path + "pseudo_uncertain.npy", np.delete(pseudo_uncertain, [0], 0))
