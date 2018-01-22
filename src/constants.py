# PATH definition
global_path = ""
initial_weights_path = "models/classical.hdf5"
final_weights_path = "models/classicalo.hdf5"

# Data definition
img_rows = 64 * 3
img_cols = 80 * 3

nb_classes = 10
nb_total = 2000
nb_train = 1600

#nb_labeled = 600
nb_labeled = nb_train
nb_unlabeled = nb_train - nb_labeled

# CEAL parameters
apply_edt = True
nb_iterations = 10
nb11 = 10
nb12 = 5
nb2 = 10
nb31 = 10
nb32 = 15
rate = 10
nb_step_predictions = 20
pseudo_epoch = 5
first_epoch = 5
second_epoch = 0
thirt_epoch = 0
nb_pseudo_initial = 20
pseudo_rate = 20

initial_train = True
apply_augmentation = False
nb_initial_epochs = 10
nb_active_epochs = 2
batch_size = 128
