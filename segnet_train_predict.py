from __future__ import print_function

import numpy as np
from keras.callbacks import ModelCheckpoint

from data import load_train_data, load_test_data
from segnet import build_model

img_rows = 256
img_cols = 256

TRAIN = 1
PREDICT = 0


def train():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = build_model()
    model.summary()
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])
    print('-' * 30)


def predict():
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_id_test = load_test_data()

    imgs_test = imgs_test.astype('float32')

    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization

    imgs_test -= mean
    imgs_test /= std

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model = build_model()
    model.load_weights('unet1.hdf5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)

    imgs_testing_train = model.predict(imgs_test, verbose=1)
    np.save('imgs_testing_train1.npy', imgs_testing_train)


if __name__ == '__main__':
    if (TRAIN):
        train()
    if (PREDICT):
        predict()
