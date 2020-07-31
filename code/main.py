"""
   Copyright 2020 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import tensorflow as tf
import configparser
import pandas as pd
import numpy as np
import tempfile
import os
from read_dataset import input_fn
from routenet_model import model_fn

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def train_and_evaluate(train_dir, eval_dir, config, model_dir=None):
    """Trains and evaluates the model.

    Args:
        train_dir (string): Path of the training directory.
        eval_dir (string): Path of the evaluation directory.
        config (configparser): Config file containing the diferent configurations
                               and hyperparameters.
        model_dir (string): Directory where all outputs (checkpoints, event files, etc.) are written.
                            If model_dir is not set, a temporary directory is used.
    """

    my_checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_secs=int(config['RUN_CONFIG']['save_checkpoints_secs']),
        keep_checkpoint_max=int(config['RUN_CONFIG']['keep_checkpoint_max'])
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=my_checkpoint_config,
        params=config
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(train_dir, repeat=True, shuffle=True),
        max_steps=int(config['RUN_CONFIG']['train_steps'])
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(eval_dir, repeat=False, shuffle=False),
        throttle_secs=int(config['RUN_CONFIG']['throttle_secs'])
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def predict(test_dir, model_dir, config):
    """Generate the predictions given a model.

    Args:
        test_dir (string): Path of the test directory.
        model_dir (string): Directory with the trained model.
        config (configparser): Config file containing the diferent configurations
                               and hyperparameters.

    Returns:
        list: A list with the predicted values.
    """

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params=config
    )

    pred_results = estimator.predict(input_fn=lambda: input_fn(test_dir, repeat=False, shuffle=False))

    return [pred['predictions'] for pred in pred_results]


def predict_and_save(test_dir, model_dir, save_dir, filename, config):
    """Generates and saves a Pandas Dataframe in CSV format with the real and the predicted delay.
    It also computes the MAPE (Mean Absolute Percentage Error) of all the samples in the dataset
    and computes its mean.

    Args:
        test_dir (string): Path of the test directory.
        model_dir (string): Directory with the trained model.
        save_dir (string): Directory where the generated dataframe will be saved (in csv).
        filename (string): The filename of the dataframe.
        config (configparser): Config file containing the diferent configurations
                               and hyperparameters.

    Returns:
        float: The Mean Absolute Percentage Error.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tmp_dir = tempfile.mkdtemp()

    ds = input_fn(test_dir, repeat=False, shuffle=False)

    dataframes_to_concat = []

    it = 0

    df_files = []
    delays = np.array([])
    for predictors, target in ds:

        it += 1
        delays = np.append(delays, target)

        if it % 1000 == 0:

            aux_df = pd.DataFrame({
                "Delay": delays
            })

            dataframes_to_concat.append(aux_df)
            delays = np.array([])

            if it % 3000 == 0:
                df = pd.concat(dataframes_to_concat)
                file = os.path.join(tmp_dir, "tmp_df_" + str(it) + ".parquet")
                df.to_parquet(file)
                df_files.append(file)
                dataframes_to_concat = []


    if it % 3000 != 0:
        if it % 1000 != 0:

            aux_df = pd.DataFrame({
                "Delay": delays
            })

            dataframes_to_concat.append(aux_df)

        df = pd.concat(dataframes_to_concat)
        file = os.path.join(tmp_dir, "tmp_df_" + str(it) + ".parquet")
        df.to_parquet(file)
        df_files.append(file)

    df_list = []

    for file in df_files:
        df_list.append(pd.read_parquet(os.path.join(file)))

    df = pd.concat(df_list)

    file = os.path.join(save_dir, filename)
    df.to_csv(file)

    predictions = predict(test_dir, model_dir, config)

    df["Predicted_Delay"] = predictions
    df['Absolute_Error'] = np.abs(df["Delay"] - df["Predicted_Delay"])
    df['Absolute_Percentage_Error'] = (df['Absolute_Error'] / np.abs(df["Delay"]))*100

    return df['Absolute_Percentage_Error'].mean()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read('../code/config.ini')

    train_and_evaluate(config['DIRECTORIES']['train'],
                       config['DIRECTORIES']['test'],
                       config._sections,
                       model_dir=config['DIRECTORIES']['logs'])

    mre = predict_and_save(config['DIRECTORIES']['test'],
                           config['DIRECTORIES']['logs'],
                           '../dataframes/',
                           'predictions.csv',
                           config._sections)
