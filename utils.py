import numpy as np


def entropy_rank(pred):
    en = np.zeros(len(pred))

    for i in range(0, len(pred)):
        en[i] = sum(-pred[i] * np.log(pred[i]))

    return np.argsort(en)[::-1]


def uncertain_set(en, nb_annotations):
    return en[0:nb_annotations]


def certain_set(en, thresh, initial_decay_rate, decay_rate):
    # Threshold updating <-- review
    if thresh == None:
        thresh = max(en) - ((max(en) - min(en)) * initial_decay_rate)
    else:
        thresh = thresh + (max(en) - thresh) * decay_rate

    return np.where(en < thresh)[0], thresh


def predictions_max_class(array, predictions, nb_classes):
    max_class = np.zeros([len(array), nb_classes])

    for i in range(0, len(array)):
        max_class[i][np.argmax(predictions[array[i]])] = 1

    return max_class