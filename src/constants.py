# PATH definition
global_path = "[global_path_name]"
initial_weights_path = "models/[initial_weights_name].hdf5"
final_weights_path = "models/[output_weights_name].hdf5"

# Data definition
img_rows = 64 * 3
img_cols = 80 * 3

nb_total = 2000
nb_train = 1600
nb_labeled = 600
nb_unlabeled = nb_train - nb_labeled

# CEAL parameters
apply_edt = True
nb_iterations = 10

nb_step_predictions = 20

nb_no_detections = 10
nb_random = 15
nb_most_uncertain = 10
most_uncertain_rate = 5

pseudo_epoch = 5
nb_pseudo_initial = 20
pseudo_rate = 20

initial_train = True
apply_augmentation = False
nb_initial_epochs = 10
nb_active_epochs = 2
batch_size = 128
