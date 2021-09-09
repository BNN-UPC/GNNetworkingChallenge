import os

# Uncomment this line in case you want to disable GPU execution
# Note you need to have CUDA installed to run de execution in GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from glob import iglob
import configparser
from itertools import zip_longest
import zipfile

from read_dataset import input_fn
from routenet_model import RouteNetModel


FILENAME = 'upload_file'

# Remember to change this path if you want to make predictions on the final
# test dataset -> './utils/paths_per_sample_test_dataset.txt'
PATHS_PER_SAMPLE = './utils/paths_per_sample_toy_dataset.txt'


##########################
#### PREDICTING BLOCK ####
##########################
def transformation(x, y):
    """Apply an intial transformation on all the samples of the dataset (before feeding the model).
       Note that here you should use the same transformation used for the model training (e.g., in <path>/code/main.py)
           Args:
               x (dict): predictor variables.
               y (array): target variable.
           Returns:
               x,y: The modified predictor/target variables.
    """
    return x, y

# Read the config.ini file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')

# Load the model
model = RouteNetModel(config)

# Load the last model checkpoint. Note that here you may not want to load the last trained model,
# but the one that obtained better performance on the validation dataset
# The path to the model is expected to be in the 'config.ini' file, variable 'logs'
ckpt_dir = config['DIRECTORIES']['logs']
latest = tf.train.latest_checkpoint(ckpt_dir)
model.load_weights(latest)

# Ensure that directories are loaded in a given order. It is IMPORTANT to keep this, as it ensures that samples
# are loaded in the desired order
directories = [d for d in iglob(config['DIRECTORIES']['test'] + '/*/*')]
# First, sort by scenario and second, by topology size
directories.sort(key=lambda f: (os.path.dirname(f), int(os.path.basename(f))))

upload_file = open(FILENAME+'.txt', "w")

predictions = []
first = True
print('Starting predictions...')
for d in directories:
    print('Current directory: ' + d)

    # It is NECESSARY to keep shuffle as 'False', as samples have to be read always in the same order
    ds_test = input_fn(d, shuffle=False)
    ds_test = ds_test.map(lambda x, y: transformation(x, y))
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # Generate predictions
    pred = model.predict(ds_test)

    # If you need to denormalize or process the model predictions do it here
    # E.g.:
    # y = np.exp(pred)

    # Separate predictions of each sample; each line contains all the per-path predictions of that sample
    # excluding those paths with no traffic (i.e., flow['AvgBw'] != 0 and flow['PktsGen'] != 0)
    idx = 0
    for x, y in ds_test:
        top_pred = pred[idx: idx+int(x['n_paths'])]
        idx += int(x['n_paths'])
        if not first:
            upload_file.write("\n")
        upload_file.write("{}".format(';'.join([format(i,'.6f') for i in np.squeeze(top_pred)])))
        first = False

upload_file.close()

zipfile.ZipFile(FILENAME+'.zip', mode='w').write(FILENAME+'.txt')

########################################################
###### CHECKING THE FORMAT OF THE SUBMISSION FILE ######
########################################################
sample_num = 0
error = False
print("Checking the file...")

with open(FILENAME+'.txt', "r") as uploaded_file, open(PATHS_PER_SAMPLE, "r") as path_per_sample:
    # Load all files line by line (not at once)
    for prediction, n_paths in zip_longest(uploaded_file, path_per_sample):
        # Case 1: Line Count does not match.
        if n_paths is None:
            print("WARNING: File must contain 1560 lines in total for the final test datset (90 for the toy dataset). "
                  "Looks like the uploaded file has {} lines".format(sample_num))
            error = True
        if prediction is None:
            print("WARNING: File must have 1560 lines in total for the final test datset (90 for the toy dataset). "
                  "Looks like the uploaded file has {} lines".format(sample_num))
            error = True

        # Remove the \n at the end of lines
        prediction = prediction.rstrip()
        n_paths = n_paths.rstrip()

        # Split the line, convert to float and then, to list
        prediction = list(map(float, prediction.split(";")))

        # Case 2: Wrong number of predictions in a sample
        if len(prediction) != int(n_paths):
            print("WARNING in line {}: The line should have size {} but it has size {}".format(sample_num, n_paths,
                                                                                               len(prediction)))
            error = True

        sample_num += 1

if not error:
    print("Congratulations! The submission file has passed all the tests!")
