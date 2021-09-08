import os

# Uncomment this line in case you want to disable GPU execution
# Note you need to have CUDA installed to run de execution in GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import ignnition
import numpy as np
from glob import iglob
from itertools import zip_longest

FILENAME = 'upload_file.txt'

# Remember to change this path if you want to make predictions on the final
# test dataset -> './utils/paths_per_sample_test_dataset.txt'
PATHS_PER_SAMPLE = './utils/paths_per_sample_toy_dataset.txt'

##########################
#### PREDICTING BLOCK ####
##########################
model = ignnition.create_model(model_dir='./')
model.computational_graph()

# Ensure that directories are loaded in a given order. It is IMPORTANT to keep this, as it ensures that samples
# are loaded in the desired order
directories = [d for d in iglob('./sample_data/test/*/*')]

# First, sort by scenario and second, by topology size
directories.sort(key=lambda f: (os.path.dirname(f), int(os.path.basename(f))))

upload_file = open(FILENAME, "w")

predictions = model.predict()
# If you need to denormalize or process the model predictions do it here
# E.g.:
# predictions = np.exp(predictions)

first = True
# Separate predictions of each sample; each line contains all the per-path predictions of that sample
# excluding those paths with no traffic (i.e., flow['AvgBw'] != 0 and flow['PktsGen'] != 0)
for pred in predictions:
    if not first:
        upload_file.write("\n")
    upload_file.write("{}".format(';'.join([str(i) for i in np.squeeze(pred)])))
    first = False

upload_file.close()

########################################################
###### CHECKING THE FORMAT OF THE SUBMISSION FILE ######
########################################################
sample_num = 0
error = False
print("Checking the file...")

with open(FILENAME, "r") as uploaded_file, open(PATHS_PER_SAMPLE, "r") as path_per_sample:
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
