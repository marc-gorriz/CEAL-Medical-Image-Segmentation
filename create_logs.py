from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from data import load_train_data
from unet import *
from utils import *

X_train, y_train = load_train_data()
X_train = preprocess(X_train)
y_train = preprocess(y_train)
X_train = X_train.astype('float32')
mean = np.mean(X_train)  # mean for data centering
std = np.std(X_train)  # std for data normalization

X_train -= mean
X_train /= std

y_train = y_train.astype('float32')
y_train /= 255.  # scale masks to [0, 1]
y_train = y_train.astype('uint8')

# DB definition

X_labeled_train = X_train[0:nb_labeled]
y_labeled_train = y_train[0:nb_labeled]
X_unlabeled_train = X_train[nb_labeled:len(X_train)]
oracle = y_train[nb_labeled:len(X_train)]


for step in range(0,nb_iterations):
    step_path = path_log + str(step) + "/"

    oracle_index = np.load(step_path + "oracle_index.npy")
    pseudo_index = np.load(step_path + "pseudo_index.npy")
    oracle_uncertain = np.load(step_path + "oracle_uncertain.npy")
    pseudo_uncertain = np.load(step_path + "pseudo_uncertain.npy")
    oracle_predictions = np.load(step_path + "oracle_predictions.npy")
    pseudo_predictions = np.load(step_path + "pseudo_predictions.npy")

    # oracle plots
    for s in range(0,len(oracle_index)):
      plt.subplots()
      plt.suptitle("Oracle image "+str(s))
      
      plt.subplot(221)
      plt.title("image")
      plt.imshow(X_unlabeled_train[oracle_index[s]].reshape([img_rows,img_cols]),cmap="gray")

      plt.subplot(222)
      plt.title("ground truth")
      plt.imshow(oracle[oracle_index[s]].reshape([img_rows, img_cols]),cmap="gray")

      plt.subplot(223)
      plt.title("prediction")
      plt.imshow(oracle_predictions[s].reshape([img_rows,img_cols]),cmap="gray")

      plt.subplot(224)
      plt.title("uncertain")
      plt.imshow(oracle_uncertain[s],cmap="gray")

      plt.savefig(step_path+"log_oracle"+str(step)+str(s))
      plt.close()   

    # pseudo plots
    for s in range(0,len(pseudo_index)):
      plt.subplots()
      plt.suptitle("Pseudo image "+str(s))

      plt.subplot(221)
      plt.title("image")
      plt.imshow(X_unlabeled_train[pseudo_index[s]].reshape([img_rows,img_cols]),cmap="gray")

      plt.subplot(222)
      plt.title("ground truth")
      plt.imshow(oracle[pseudo_index[s]].reshape([img_rows, img_cols]),cmap="gray")

      plt.subplot(223)
      plt.title("prediction")
      plt.imshow(pseudo_predictions[s].reshape([img_rows,img_cols]),cmap="gray")

      plt.subplot(224)
      plt.title("uncertain")
      plt.imshow(pseudo_uncertain[s],cmap="gray")

      plt.savefig(step_path+"log_pseudo"+str(step)+str(s))
      plt.close()
