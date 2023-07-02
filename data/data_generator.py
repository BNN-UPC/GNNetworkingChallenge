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

# Only run as main
if __name__ != "__main__":
    raise RuntimeError("This script should not be imported!")

# Parse imports
from typing import Tuple, Generator, Dict, Any, List
import numpy as np
import tensorflow as tf
from itertools import permutations
from re import sub
import argparse
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n-folds", type=int, default=5)
args = parser.parse_args()

# import datasets's DataNet API
sys.path.insert(0, args.input_dir)
from datanetAPI import DatanetAPI, TimeDist, Sample


def _get_network_decomposition(sample: Sample) -> Tuple[dict, list]:
    """Given a sample from the DataNet API, it returns it as a sample for the model.

    Parameters
    ----------
    sample : DatanetAPI.Sample
        Sample from the DataNet API

    Returns
    -------
    Tuple[dict, list]
        Tuple with the inputs of the model and the target variable to predict

    Raises
    ------
    ValueError
        Raised if one of the links is of an unknown type
    """

    # Read values from the DataNet API
    network_topology = sample.get_physical_topology_object()
    traffic_matrix = sample.get_traffic_matrix()
    physical_path_matrix = sample.get_physical_path_matrix()
    performance_matrix = sample.get_performance_matrix()
    # Process sample id (for debugging purposes and for then evaluating the model)
    sample_file_path, sample_file_id = sample.get_sample_id()
    sample_file_name = sample_file_path.split("/")[-1]
    # Obtain links and nodes
    # We discard all links that start from the traffic generator
    links = dict()
    for edge in network_topology.edges:  # src, dst, port
        # We identify all traffic generators as the same port
        edge_id = sub(r"t(\d+)", "tg", network_topology.edges[edge]["port"])
        if edge_id.startswith("r") or edge_id.startswith("s"):
            links[edge_id] = {
                "capacity": float(network_topology.edges[edge].get("bandwidth", 1e9))
                / 1e9,  # original value is in bps, we change it to Gbps
            }
        elif edge_id.startswith("tg"):
            continue
        else:
            raise ValueError(f"Unknown edge type: {edge_id}")

    # In this scenario assume that flows can either follow CBR or MB distributions
    flows = dict()
    used_links = set()  # Used later so we only consider used links
    # Add flows
    for src, dst in filter(
        lambda x: traffic_matrix[x]["AggInfo"]["AvgBw"] != 0
        and traffic_matrix[x]["AggInfo"]["PktsGen"] != 0,
        permutations(range(len(traffic_matrix)), 2),
    ):
        for local_flow_id in range(len(traffic_matrix[src, dst]["Flows"])):
            flow = traffic_matrix[src, dst]["Flows"][local_flow_id]
            # Size distribution is always determinstic
            # Obtain and clean the path followed the flow
            # We must also clean up the name of the traffic generator
            clean_og_path = [
                sub(r"t(\d+)", "tg", link)
                for link in physical_path_matrix[src, dst][2::2]
            ]
            flow_id = f"{src}_{dst}_{local_flow_id}"
            flows[flow_id] = {
                "source": src,
                "destination": dst,
                "flow_id": flow_id,
                "length": len(clean_og_path),
                "og_path": clean_og_path,
                "traffic": flow["AvgBw"],  # in bps
                "packets": flow["PktsGen"],
                "packet_size": flow["SizeDistParams"]["AvgPktSize"],
                "flow_type": (
                    float(flow["TimeDist"] == TimeDist.CBR_T),
                    float(flow["TimeDist"] == TimeDist.MULTIBURST_T),
                ),
                "delay": performance_matrix[src, dst]["Flows"][local_flow_id][
                    "AvgDelay"
                ]
                * 1000,  # in ms
            }

            # Add edges to the used_links set
            used_links.update(set(clean_og_path))

    # Purge unused links
    links = {kk: vv for kk, vv in links.items() if kk in used_links}

    # Normalize flow naming
    # We give the indices in such a way that flows states are concatanated as [CBR, MB]
    ordered_flows = list()
    flow_mapping = dict()
    for idx, (flow_id, flow_params) in enumerate(flows.items()):
        flow_mapping[flow_id] = idx
        ordered_flows.append(flow_params)
    n_f = len(ordered_flows)

    # Normalize link naming
    ordered_links = list()
    link_mapping = dict()
    for idx, (link_id, link_params) in enumerate(links.items()):
        link_mapping[link_id] = idx
        ordered_links.append(link_params)
    n_l = len(ordered_links)

    # Obtain list of indices representing the topology
    # link_to_path: two dimensional array, first dimension are the paths, second dimension are the link indices
    link_to_path = list()
    # We define link_pos_in_flows that will later help us build path_to_link
    link_pos_in_flows = list()
    for og_path in map(lambda x: x["og_path"], ordered_flows):
        # This list will contain the link indices in the original path,in order
        local_list = list()
        # This dict indicates for each link which are the positions in the original path, if any
        local_dict = dict()
        for link_id in og_path:
            # Transform link_id into a link index
            link_idx = link_mapping[link_id]
            local_dict.setdefault(link_idx, list()).append(len(local_list))
            local_list.append(link_idx)
        link_to_path.append(local_list)
        link_pos_in_flows.append(local_dict)

    # path_to_link: two dimensional array, first dimension are the links, second dimension are tuples.
    # Each tuple contains the path index and the link's position in the path
    # Note that a link can appear in multiple paths and multiple times in the same path
    path_to_link = list()
    for link_idx in range(n_l):
        local_list = list()
        for flow_idx in range(n_f):
            if link_idx in link_pos_in_flows[flow_idx]:
                local_list += [
                    (flow_idx, pos) for pos in link_pos_in_flows[flow_idx][link_idx]
                ]
        path_to_link.append(local_list)

    # Many of the features must have expanded dimensions so they can be concatenated
    sample = (
        {
            # Identifier features
            "sample_file_name": [sample_file_name] * n_f,
            "sample_file_id": [sample_file_id] * n_f,
            "flow_id": [flow["flow_id"] for flow in ordered_flows],
            # Flow attributes
            "flow_traffic": np.expand_dims(
                [flow["traffic"] for flow in ordered_flows], axis=1
            ),
            "flow_packets": np.expand_dims(
                [flow["packets"] for flow in ordered_flows], axis=1
            ),
            "flow_packet_size": np.expand_dims(
                [flow["packet_size"] for flow in ordered_flows], axis=1
            ),
            "flow_type": [flow["flow_type"] for flow in ordered_flows],
            "flow_length": [flow["length"] for flow in ordered_flows],
            # Link attributes
            "link_capacity": np.expand_dims(
                [link["capacity"] for link in ordered_links], axis=1
            ),
            # Topology attributes
            "link_to_path": tf.ragged.constant(link_to_path),
            "path_to_link": tf.ragged.constant(path_to_link, ragged_rank=1),
        },
        [flow["delay"] for flow in ordered_flows],
    )

    return sample


def _generator(
    data_dir: str, shuffle: bool
) -> Generator[Tuple[Dict[str, Any], List[float]], None, None]:
    """Returns processed samples from the given dataset.

    Parameters
    ----------
    data_dir : str
        Path to the dataset
    shuffle : bool
        True to shuffle the samples, False otherwise

    Yields
    ------
    Generator[Tuple[Dict[str, Any], List[float]], None, None]
        Returns a generator of tuples, where the first element is a dictionary with the sample's features
        and the second element is a list of the sample's labels (in this case, the flow's delay)
    """
    try:
        data_dir = data_dir.decode("UTF-8")
    except (UnicodeDecodeError, AttributeError):
        pass
    tool = DatanetAPI(data_dir, shuffle=shuffle)
    for sample in iter(tool):
        ret = _get_network_decomposition(sample)
        # SKIP SAMPLES WITH ZERO OR NEGATIVE VALUES
        if not all(x > 0 for x in ret[1]):
            continue
        yield ret


def input_fn(data_dir: str, shuffle: bool = False) -> tf.data.Dataset:
    """Returns a tf.data.Dataset object with the dataset stored in the given path

    Parameters
    ----------
    data_dir : str
        Path to the dataset
    shuffle : bool, optional
        True to shuffle the samples, False otherwise, by default False

    Returns
    -------
    tf.data.Dataset
        The processed dataset
    """
    signature = (
        {
            "sample_file_name": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "sample_file_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "flow_id": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "flow_traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_packets": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_packet_size": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_type": tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            "flow_length": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "link_capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "link_to_path": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
            "path_to_link": tf.RaggedTensorSpec(
                shape=(None, None, 2), dtype=tf.int32, ragged_rank=1
            ),
        },
        tf.TensorSpec(shape=None, dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        _generator,
        args=[data_dir, shuffle],
        output_signature=signature,
    )

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


# MAIN: generate the dataset

# Set seeds for reproducibility
np.random.seed(args.seed)
random.seed(args.seed)
tf.random.set_seed(args.seed)

# Parse dataset
tf.data.Dataset.save(
    input_fn(
        args.input_dir,
        shuffle=args.shuffle,
    ),
    args.output_dir,
    compression="GZIP",
)

# Split dataset into 5 folds
# NOTE: the reason why we store the dataset and load it again is because otherwise
# tensorflow will try regenerate the dataset using the generator for each fold
# which results in taking a lot of time
ds = tf.data.Dataset.load(args.output_dir, compression="GZIP")
ds_split = [ds.shard(args.n_folds, ii) for ii in range(args.n_folds)]
dataset_name = args.output_dir if args.output_dir[-1] != "/" else args.output_dir[:-1]

for ii in range(args.n_folds):
    # Split dataset into train and validation
    tr_splits = [jj for jj in range(args.n_folds) if jj != ii]
    ds_val = ds_split[ii]
    ds_train = ds_split[tr_splits[0]]
    for jj in tr_splits[1:]:
        ds_train = ds_train.concatenate(ds_split[jj])

    # Save datasets
    tf.data.Dataset.save(
        ds_train, f"{dataset_name}_cv/{ii}/training", compression="GZIP"
    )
    tf.data.Dataset.save(
        ds_val, f"{dataset_name}_cv/{ii}/validation", compression="GZIP"
    )
