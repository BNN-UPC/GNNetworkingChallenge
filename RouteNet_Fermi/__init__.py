from sys import stderr
import warnings

warnings.filterwarnings("ignore")
seed_value = 69420
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYTHONHASHSEED'] = str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.random.set_seed(seed_value)

tf.get_logger().setLevel('INFO')

from .data_generator import input_fn
from .model import RouteNet_Fermi


def main(train_path, final_evaluation=False, ckpt_dir="./modelCheckpoints"):
    """
    Trains and evaluates the model with the provided dataset.
    The model will be trained for 20 epochs.
    At each epoch a checkpoint of the model will be generated and stored at the folder ckpt_dir which will be created automatically if it doesn't exist already.
    Training the model will also generate logs at "./logs" that can be opened with tensorboard.

    Parameters
    ----------
    train_path
        Path to the training dataset
    final_evaluation, optional
        If True after training the model will be validated using all of the validation dataset, by default False
    ckpt_dir, optional
        Relative path (from the repository's root folder) where the model's weight will be stored, by default "./modelCheckpoints"
    """

    if (not os.path.exists(train_path)):
        print(f"ERROR: the provided training path \"{os.path.abspath(train_path)}\" does not exist!", file=stderr)
        return None
    TEST_PATH = './validation_dataset'
    if (not os.path.exists(TEST_PATH)):
        print("ERROR: Validation dataset not found at the expected location:",
              os.path.abspath(TEST_PATH), file=stderr)
        return None
    LOG_PATH = './logs'
    if (not os.path.exists(LOG_PATH)):
        print("INFO: Logs folder created at ", os.path.abspath(LOG_PATH))
        os.makedirs(LOG_PATH)
    # Check dataset size
    dataset_size = len([0 for _ in input_fn(train_path, shuffle=True)])
    if not dataset_size:
        print(f"ERROR: The dataset has no valid samples!", file=stderr)
        return None
    elif (dataset_size > 100):
        print(f"ERROR: The dataset can only have up to 100 samples (currently has {dataset_size})!", file=stderr)
        return None

    ds_train = input_fn(train_path, shuffle=True, training=True)
    ds_train = ds_train.repeat()
    ds_test = input_fn(TEST_PATH, shuffle=False)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    model = RouteNet_Fermi()

    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(loss=loss_object,
                  optimizer=optimizer,
                  run_eagerly=False)

    model.load_weights('./RouteNet_Fermi/initial_weights/initial_weights')

    latest = tf.train.latest_checkpoint(ckpt_dir)

    if latest is not None:
        print(f"ERROR: Found a pretrained models, please clear or remove the {ckpt_dir} directory and try again!")
        return None
    else:
        print("INFO: Starting training from scratch...")

    filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        verbose=1,
        mode="min",
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True,
        save_freq='epoch')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH, histogram_freq=1)

    model.fit(ds_train,
              epochs=20,
              steps_per_epoch=2000,
              validation_data=ds_test,
              validation_steps=20,
              callbacks=[cp_callback, tensorboard_callback],
              use_multiprocessing=True)

    if final_evaluation:
        print("Final evaluation:")
        model.evaluate(ds_test)


def evaluate(ckpt_path):
    """
    Loads model from checkpoint and trains the model.

    Parameters
    ----------
    ckpt_path
        Path to the checkpoint. Format the name as it was introduced in tf.keras.Model.load_weights.
    """

    TEST_PATH = './validation_dataset'
    if (not os.path.exists(TEST_PATH)):
        print("ERROR: Validation dataset not found at the expected location:",
              os.path.abspath(TEST_PATH), file=stderr)
        return None

    ds_test = input_fn(TEST_PATH, shuffle=False)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    model = RouteNet_Fermi()

    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(loss=loss_object,
                  optimizer=optimizer,
                  run_eagerly=False)

    # stored_weights = tf.train.load_checkpoint(ckpt_path)
    model.load_weights(ckpt_path)

    # Evaluate model
    model.evaluate(ds_test)
