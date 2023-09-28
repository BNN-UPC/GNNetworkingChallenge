"""
   Copyright 2023 Universitat Politècnica de Catalunya

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
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from typing import Optional, Tuple, Callable
import pandas as pd
import numpy as np


def print_err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _default_individual_prediction(model: tf.keras.Model, sample: any) -> np.ndarray:
    """
    Default function to predict the flow delay values for a given sample. The prediction
    is returned as a flat, unnormalized numpy array and in seconds.

    Parameters
    ----------
    model : tf.keras.Model
        The model to use for prediction. Its weights should be already trained.
    sample_features : any
        The sample to predict. By default it expects samples from a tf.data.Dataset,
        but can be any format that allows to iterate over it.


    Returns
    -------
    np.ndarray
        The predictions to return.
    """
    # Obtain the prediction as numpy array, and flatten
    pred = model(sample).numpy().reshape((-1,))
    # Transform the prediction from ms to s, and return
    return  pred / 1000


def predict(
    ds: any,
    model: tf.keras.Model,
    predict_file_name: str,
    submission_verification_file_path: Optional[
        str
    ] = "verification_files/submission_verification.txt",
    predict_file_path: Optional[str] = None,
    individual_prediction: Callable[[any, any], np.ndarray] = _default_individual_prediction,
    verbose: bool = False,
) -> None:
    """Use the given model to predict flow delay values for the given dataset.
    Predictions are stored in a csv file, which itself will be compressed with zip.

    Parameters
    ----------
    ds : any
        Object representing the loaded test dataset. By default it expects the
        tf.data.Dataset format, but can be any format that allows to iterate over it.

    model : tf.keras.Model
        tf.keras.Model instance to use for prediction. Its weights should be already
        trained.

    predict_file_name : str,
        Name of the generated file with all the predictions

    submission_verification_file_path : Optional[str], optional
        Path to the file which contains the data to verify the submission.
        This file is provided by the challenge. There is one version of the file for
        the toy dataset, and another one for test dataset. By default, it points to the
        test dataset verification file.

    predict_file_path : Optional[str], optional
        Path to the directory where to store the predictions. By default, the prediction
        are stored in the current directory.

    verbose : bool, optional
        If True, print additional information, by default False
    """
    list_sample_file_id = []
    list_flow_id = []
    list_predicted_delay = []
    num_sample_file_id = []
    num_flow_id = []
    num_predicted_delay = []
    if verbose:
        print()
    for ii, (sample_features, _) in enumerate(iter(ds)):
        if verbose:
            print(f"\r Progress: {ii} / {len(ds)}", end="")

        # For each sample, fill in the fields
        sample_file_id = sample_features["sample_file_id"].numpy().tolist()
        list_sample_file_id += sample_file_id
        num_sample_file_id.append(len(sample_file_id))

        flow_id = list(map(lambda x: x.decode(), sample_features["flow_id"].numpy()))
        list_flow_id += flow_id
        num_flow_id.append(len(flow_id))

        predicted_delay = individual_prediction(model, sample_features).tolist()
        list_predicted_delay += predicted_delay
        num_predicted_delay.append(len(predicted_delay))
    if verbose:
        print()

    # Verify the submission
    with open(submission_verification_file_path, "r") as f:
        flows_per_sample = list(map(lambda x: int(x.strip()), f.readlines()))
        total_flows = sum(flows_per_sample)
    success = True

    # 1. Check total number of flows is correct
    if len(list_sample_file_id) != total_flows:
        print_err(
            f"ERROR: When counting the number of sample files id, the number of flows "
            + f"is incorrect. Expected {total_flows}, got {len(list_sample_file_id)}"
        )
        success = False
    if len(list_flow_id) != total_flows:
        print_err(
            f"ERROR: When counting the number of flows id, the number of flows is "
            + f"incorrect. Expected {total_flows}, got {len(list_flow_id)}"
        )
        success = False
    if len(list_predicted_delay) != total_flows:
        print_err(
            f"ERROR: When counting the number of predicted delays, the number of flows "
            + f"is incorrect. Expected {total_flows}, got {len(list_predicted_delay)}"
        )
        success = False

    # 2. Check the number of flows per sample is correct
    for ii, (num_sample, num_flow, num_delay, true_flows) in enumerate(
        zip(num_sample_file_id, num_flow_id, num_predicted_delay, flows_per_sample)
    ):
        num_sample_ver = num_sample != true_flows
        num_flow_ver = num_flow != true_flows
        num_delay_ver = num_delay != true_flows

        if num_sample_ver or num_flow_ver or num_delay_ver:
            err_msg = f"ERROR: The number of flows for sample {ii} is incorrect."
            if num_sample_ver:
                err_msg += f" Expected {true_flows} sample file ids, got {num_sample}."
            if num_flow_ver:
                err_msg += f" Expected {true_flows} flow ids, got {num_flow}."
            if num_delay_ver:
                err_msg += f" Expected {true_flows} predicted delays, got {num_delay}."
            print_err(err_msg)
            success = False

    if success is False:
        print_err(
            f"WARNING: The submission is not correct. Please check the errors above. "
            + f"Exiting..."
        )
        sys.exit(1)
    # Save predictions using pandas
    print("Verification passed! Saving predictions...")
    df = pd.DataFrame(
        {
            "sample_file_id": list_sample_file_id,
            "flow_id": list_flow_id,
            "predicted_delay": list_predicted_delay,
        }
    )
    if predict_file_path is not None:
        os.makedirs(predict_file_path, exist_ok=True)
        zip_path = f"{os.path.join(predict_file_path, predict_file_name)}.zip"
    else:
        zip_path = f"{predict_file_name}.zip"

    df.to_csv(
        zip_path,
        index=False,
        sep=";",
        header=False,
        float_format="%.9f",
        compression={"method": "zip", "archive_name": f"{predict_file_name}.csv"},
    )


if __name__ == "__main__":
    import argparse
    import models
    from train import get_min_max_dict

    parser = argparse.ArgumentParser(
        description="Use a trained model to generate predictions from"
    )
    parser.add_argument("-ds", type=str, help="Either 'CBR+MB' or 'MB'", required=True)
    parser.add_argument(
        "--ckpt-path", type=str, help="Path to checkpoint", required=True
    )
    parser.add_argument(
        "--tr-path",
        type=str,
        help="Path to training dataset (needed to obtain normalization values)",
        required=True,
    )
    parser.add_argument(
        "--te-path",
        type=str,
        help="Path to test dataset (to generate the predictions from)",
        required=True,
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="If True, the script expects the toy dataset. The purpose of this "
        + "dataset is to test the script and ensure the model is generating correct "
        + "predictions.",
    )
    args = parser.parse_args()

    # Check the scenario
    if args.ds == "CBR+MB":
        model = models.Baseline_cbr_mb()
    elif args.ds == "MB":
        model = models.Baseline_mb()
    else:
        raise ValueError("Unrecognized dataset")

    # Compute normalization values
    model.set_min_max_scores(
        get_min_max_dict(
            tf.data.Dataset.load(args.tr_path, compression="GZIP"),
            model.min_max_scores_fields,
        )
    )
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
    )
    # Load the checkpoint
    model.load_weights(args.ckpt_path)

    # Select correct verification file
    ver_file_path = (
        "verification_files/submission_verification_toy.txt"
        if args.toy
        else "verification_files/submission_verification.txt"
    )

    # Load the test dataset
    ds = tf.data.Dataset.load(args.te_path, compression="GZIP")

    # Predict
    predict(ds, model, "predictions", ver_file_path)
