import tensorflow as tf
import configparser
import pandas as pd
import numpy as np
from read_dataset import input_fn
from routenet_model import model_fn

###################
# Input variables #
###################
# Path to the test dataset root directory
test_dataset_directory = "../data/sample_data/test"
# path to the configuration file (set correctly the paths to the trained model within this file - i.e., "logs" variable)
config_file_path = '../code/config.ini'
# The filename of the output compressed CSV file (in ZIP format)
output_filename = 'submission_file'



def generate_upload_csv(test_dir, model_dir, filename, config):
    """Generates, compresses (in ZIP) and saves a Pandas Dataframe in CSV format with the predicted delays.

    Args:
        test_dir (string): Path of the test dataset root directory.
        model_dir (string): Directory of the trained model.
        filename (string): The filename of the compressed CSV file.
        config (configparser): Config file containing the different configurations
                               and hyperparameters.
    """

    # IMPORTANT NOTE! In order to compress the data, pandas needs for the output file a simple filename, without including the route or path and the extension.
    # (i.e., "submission_file", not "./home/dataset/submission_file.zip")
    if '/' in filename:
        print("---WARNING---")
        print("---Filename must be a simple filename, it should not include a path--- Use \"submission_file\" instead of \"./home/dataset/submission_file.zip\"")

    print("GENERATING DELAY LABELS WITH THE TRAINED MODEL...")
    ########################
    # Generate predictions #
    ########################

    # Create the estimator loading the model
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=config
    )

    # Generate the dataset and make the predictions
    pred_results = estimator.predict(input_fn=lambda: input_fn(test_dir, repeat=False, shuffle=False))

    # Collect the predictions
    pred = np.array([pred['predictions'] for pred in pred_results])

    ###################
    # Denormalization #
    ###################
     # If you have applied any normalization, please denormalize the predicted values here

    ####################
    # Prepare the data #
    ####################
    print("RESHAPING THE DATA...")
    # Prepare the data as it should be in the CSV file (each line contains the 342 src-dst delays of a sample)
    # The network of the test dataset has in total 342 src-dst paths (19 sources x 18 destinations = 342 src-dst pairs)
    pred = pred.reshape(int(pred.shape[0] / 342), 342)

    print("CHECKING CSV format...")
    if pred.shape != (50000, 342):
        print("--- WARNING ---")
        print("--- The format of the CSV file is not correct. It must have 50,000 lines with 342 values each one---")
        print("It has currently the following lines and and elements: " +str(pred.shape))

    print("SAVING CSV FILE COMPRESSED IN ZIP...")

    df = pd.DataFrame(pred)
    # The CSV file will be directly compressed in ZIP
    compression_options = dict(method='zip', archive_name=f'{filename}.csv')
    # The CSV file uses ";" as separator between values
    df.to_csv(f'{filename}.zip', header=False, index=False, sep=";", compression=compression_options)


# MAIN
# Loading the Configuration file with the model hyperparameters
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read(config_file_path)


generate_upload_csv(test_dataset_directory,
                    # It loads the last saved model within the "logs" directory of the config.ini file
                    config['DIRECTORIES']['logs'],
                    output_filename,
                    config._sections)
