from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from constants import *
from unet import get_unet
import cv2

def range_transform(sample):

  if (np.max(sample)==1):
    sample = sample*255

  m = 255/(np.max(sample)-np.min(sample))
  n = 255-m*np.max(sample)
  return (m*sample+n)/255

def interval(data, start, end):
  p = np.where(data >= start)[0]
  return p[np.where(data[p] < end)[0]]
"""
def get_pseudo_index(uncertain, nb):
  h = np.histogram(uncertain, 80)

  for index in range(1,len(h[0])):
    if(h[0][index]>=h[0][0]):
      break
  pseudo = interval(uncertain, h[1][index-1], h[1][index])
  np.random.shuffle(pseudo)
  return pseudo[0:nb]



def get_pseudo_index(uncertain, nb):
  h = np.histogram(uncertain, 80)

  for index in range(1,len(h[0])):
    if(h[0][index]>=h[0][0]):
      break
  pseudo = interval(uncertain, h[1][index-1], h[1][index])
  np.random.shuffle(pseudo)
  return pseudo[0:nb]
"""

def get_pseudo_index(uncertain, nb):
  h = np.histogram(uncertain, 80)

  pseudo = interval(uncertain, h[1][np.argmax(h[0])], h[1][np.argmax(h[0])+1])
  np.random.shuffle(pseudo)
  return pseudo[0:nb]

def get_thirt_index(uncertain, nb3, iteration):
  if (iteration < thirt_epoch):
    nb3=0
  histo = np.histogram(uncertain, 80)
  index = interval(uncertain, histo[1][np.argmax(histo[0])+6], histo[1][len(histo[0])-33])
  np.random.shuffle(index)
  return index[0:nb3]

"""
def get_first_index(uncertain, nb1):
  histo = np.histogram(uncertain, 80)
  oracle = interval(uncertain, histo[1][0], histo[1][2])
  np.random.shuffle(oracle)
  return oracle[0:nb1]
"""

def get_first_index(rank, nb1):
  return rank[0:nb1]
"""
def get_first_index(rank, nb11, nb12, iteration):
  if iteration < first_epoch:
    return rank[0:nb11]
  else:
    return rank[0:nb12]
"""
def get_second_index(uncertain, nb2, rate, iteration):
  data = np.array([]).astype('int')
  if (iteration>=second_epoch):
	  histo = np.histogram(uncertain, 80)
	  print(histo[0][np.argmax(histo[0])])
	  print("rate"+str(np.floor(histo[0][np.argmax(histo[0])]*rate)))
	  p = np.arange(len(histo[0])-rate,len(histo[0])-1)
	  pr = np.argsort(histo[0][p])
	  cnt = 0
	  pos = 0
	  index = np.array([]).astype('int')
	  while(cnt<nb2 and pos<len(pr)):
	    sbin = histo[0][p[pr[pos]]]

	    index = np.append(index,p[pr[pos]])
	    cnt = cnt + sbin
	    pos = pos + 1
	  
	  
	  for i in range(0,pos):
	    data = np.concatenate((data,interval(uncertain, histo[1][index[i]], histo[1][index[i]+1])))
	  np.random.shuffle(data)
	  return data[0:nb2]
  else:
    return data  

def get_oracle_index(uncertain, rank, nb1,  nb2, nb3, rate, iteration):
  return np.concatenate((get_first_index(rank, nb1), get_second_index(uncertain, nb2, rate, iteration), get_thirt_index(uncertain, nb3, iteration)))
"""
def get_oracle_index(uncertain, rank, nb1, nb2, nb3):
  h = np.histogram(uncertain, 80)
  oracle = interval(uncertain, h[1][0], h[1][3])
  np.random.shuffle(oracle) 
  return np.concatenate((oracle[0:nb1], rank[0:nb2], get_thirt_index(uncertain, nb3)))
"""
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
        return np.sum(var*transform)

    else:
        return np.sum(np.var(X, axis=0))

def compute_dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.reshape([1,img_rows*img_cols])
    y_pred_f = y_pred.reshape([1,img_rows*img_cols])
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def compute_train_sets(X_train, y_train, labeled_index, unlabeled_index, weights, iteration):
    
    print("\nActive iteration "+str(iteration))
    print("-"*50+"\n")

    #load models
    modelUncertain = get_unet(dropout=True)
    modelUncertain.load_weights(weights)
    modelPredictions = get_unet(dropout=False)
    modelPredictions.load_weights(weights)

    #predictions
    print("Computing log predictions ...\n")
    predictions = predict(X_train[unlabeled_index], modelPredictions)

    uncertain = np.zeros(len(unlabeled_index))
    accuracy = np.zeros(len(unlabeled_index))

    print("Computing train sets ...")
    for index in range(0, len(unlabeled_index)):

        if index % 100 == 0:       
            print("completed: "+str(index)+"/"+str(len(unlabeled_index)))

        sample = X_train[unlabeled_index[index]].reshape([1,1,img_rows, img_cols])

        sample_prediction = cv2.threshold(predictions[index],0.5,1,cv2.THRESH_BINARY)[1].astype('uint8')
        sample_truth = y_train[unlabeled_index[index]][0]
        #sample_truth.astype('uint8') (?)
        accuracy[index] = compute_dice_coef(y_train[unlabeled_index[index]][0], sample_prediction)
        uncertain[index] = compute_uncertain(sample, sample_prediction, modelUncertain)
    
    rank = np.argsort(uncertain)

    np.save(global_path+"logs/uncertain"+str(iteration), uncertain)
    np.save(global_path+"logs/accuracy"+str(iteration), accuracy)

    #TODO: only change oracle/pseudo index !
    if (iteration < first_epoch):
      nb1 = nb11
      nb3 = nb31
    else:
      nb1 = nb12
      nb3 = nb32
    oracle_index = get_oracle_index(uncertain, rank, nb1,  nb2, nb3, rate, iteration)
    #oracle_index = get_oracle_index(uncertain)
    oracle_rank = unlabeled_index[oracle_index]

    np.save(global_path+"ranks/oracle"+str(iteration), oracle_rank)
    np.save(global_path+"ranks/oraclelogs"+str(iteration), oracle_index)
    
    labeled_index = np.concatenate((labeled_index, oracle_rank))
    
    if(iteration >= pseudo_epoch):

      pseudo_index = get_pseudo_index(uncertain, nb_pseudo_initial + (pseudo_rate * (iteration - pseudo_epoch) ) )
      pseudo_rank = unlabeled_index[pseudo_index]
      
      np.save(global_path+"ranks/pseudo"+str(iteration), pseudo_rank)
      np.save(global_path+"ranks/pseudologs"+str(iteration), pseudo_index)
      
      X_labeled_train = np.concatenate((X_train[labeled_index], X_train[pseudo_index]))
      y_labeled_train = np.concatenate((y_train[labeled_index], predictions[pseudo_index]))
    
    else: 
      X_labeled_train = np.concatenate((X_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])
      y_labeled_train = np.concatenate((y_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])
    
    #TODO: change (test without pseudo labels)
    #X_labeled_train = np.concatenate((X_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])
    #y_labeled_train = np.concatenate((y_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])

    unlabeled_index = np.delete(unlabeled_index, oracle_index, 0)
    
    return X_labeled_train, y_labeled_train, labeled_index, unlabeled_index

"""
def data_generator():
    return ImageDataGenerator(
       featurewise_center=True,
       featurewise_std_normalization=True,
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
       horizontal_flip=True)
"""
def data_generator():
    return ImageDataGenerator(
       featurewise_center=True,
       featurewise_std_normalization=True,
       width_shift_range=0.2,
       rotation_range = 40,
       horizontal_flip=True)

def log(history, step, log_file):
    for i in range(0, len(history.history["loss"])):
        if len(history.history.keys()) == 4:
            log_file.write('{0} {1} {2} {3} \n'.format(str(step), str(i), str(history.history["loss"][i]),
                                                       str(history.history["val_dice_coef"][i])))

def create_paths():
    path_ranks = global_path+"ranks/"
    path_logs = global_path+"logs/"
    path_plots = global_path+"plots/"
    path_models = global_path+"models/"

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


