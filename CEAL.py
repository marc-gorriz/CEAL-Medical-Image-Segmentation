from __future__ import print_function

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from data import load_train_data
from unet import *
from utils import *
from constants import *

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

#Data augmentation

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# (1) Initialize model

model = get_unet(dropout=True)
weights = final_weights_path
model_checkpoint = ModelCheckpoint(weights, monitor='loss', save_best_only=True)

if(initial_train):

  if(augmentation):
    model.fit_generator(datagen.flow(X_labeled_train, y_labeled_train, batch_size=32, shuffle=True),
                    steps_per_epoch=len(X_labeled_train), nb_epoch=nb_initial_epochs, 
                    verbose=1, callbacks=[model_checkpoint])
  else:
    model.fit(X_labeled_train, y_labeled_train, batch_size=32, nb_epoch=nb_initial_epochs, verbose=1,
              shuffle=True, callbacks=[model_checkpoint])

else:
  weights = initial_weights_path
  model.load_weights(initial_weights_path)

# Active loop

for iteration in range(0, nb_iterations):

    if iteration > 0 and initial_train:
      weights = final_weights_path

    # (2) Labeling
    
    predictions_rank = uncertain_rank(X_unlabeled_train, nb_step_predictions, uncertain_method, weights)
    
    X_oracle_train = X_unlabeled_train[predictions_rank[0:nb_annotations]]
    y_oracle_train = oracle[predictions_rank[0:nb_annotations]]

    X_pseudo_train = X_unlabeled_train[predictions_rank[len(predictions_rank) - nb_pseudo:len(predictions_rank)]]
    y_pseudo_train = predict(X_pseudo_train, weights)

    np.save("x"+str(iteration),X_oracle_train)
    np.save("y"+str(iteration),y_oracle_train)

    X_labeled_train_aux = np.concatenate((X_labeled_train, X_oracle_train, X_pseudo_train))
    y_labeled_train_aux = np.concatenate((y_labeled_train, y_oracle_train, y_pseudo_train))

    save_logs(predictions_rank, X_unlabeled_train, iteration, weights)

    # (3) Training

    model.fit(X_labeled_train_aux, y_labeled_train_aux, batch_size=32, nb_epoch=nb_active_epochs, verbose=1,
              shuffle=True, callbacks=[model_checkpoint])


    X_labeled_train = np.concatenate((X_labeled_train, X_oracle_train))
    y_labeled_train = np.concatenate((y_labeled_train, y_oracle_train))
    X_unlabeled_train = np.delete(X_unlabeled_train, predictions_rank[0:nb_annotations], 0)
    oracle = np.delete(oracle, predictions_rank[0:nb_annotations], 0)
