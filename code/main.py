"""
   Copyright 2020 Universitat Politècnica de Catalunya & AGH University of Science and Technology

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
        input_fn=lambda: input_fn(train_dir),
        max_steps=int(config['RUN_CONFIG']['train_steps'])
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(eval_dir, repeat=False, shuffle=False),
        throttle_secs=int(config['RUN_CONFIG']['throttle_secs'])
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def predict(test_dir, model_dir):
    """Generate the predictions given a model.

    Args:
        test_dir (string): Path of the test directory.
        model_dir (string): Directory with the trained model.

    Returns:
        list: A list with the predicted values.
    """

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir
    )

    pred_results = estimator.predict(input_fn=lambda: input_fn(test_dir, repeat=False, shuffle=False))

    return [pred['predictions'] for pred in pred_results]

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read('../code/config.ini')

    train_and_evaluate(config['DIRECTORIES']['train'],
                       config['DIRECTORIES']['test'],
                       config._sections,
                       model_dir=config['DIRECTORIES']['logs'])