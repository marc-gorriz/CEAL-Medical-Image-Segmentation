# PATH definition
initial_weights_path = "models/unet_augment10epoch.hdf5"
final_weights_path = "models/unet_augment10epoch_test1.hdf5"

# Data definition
img_rows = 64 * 3
img_cols = 80 * 3

nb_classes = 10
nb_train = 2000

nb_labeled = 600
nb_unlabeled = nb_train - nb_labeled

# CEAL parameters
uncertain_method = "variance"
apply_edt = True
nb_iterations = 3
nb_annotations = 10
nb_pseudo = 100
nb_step_predictions = 10

initial_train = False
nb_initial_epochs = 5
nb_active_epochs = 1
batch_size = 128

#logs
path_log = "only_oracle_aaug/"
sel_random = False
nb_log_pseudo = 10
