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

import numpy as np
import tensorflow as tf

from datanetAPI import DatanetAPI

POLICIES = np.array(['WFQ', 'SP', 'DRR'])

def generator(data_dir, shuffle = False):
    """This function uses the provided API to read the data and returns
       and returns the different selected features.

    Args:
        data_dir (string): Path of the data directory.
        shuffle (string): If true, the data is shuffled before being processed.

    Returns:
        tuple: The first element contains a dictionary with the following keys:
            - bandwith
            - packets
            - link_capacity
            - links
            - paths
            - sequences
            - n_links, n_paths
            The second element contains the source-destination delay
    """
    tool = DatanetAPI(data_dir, [], shuffle)
    it = iter(tool)
    for sample in it:
        ###################
        #  EXTRACT PATHS  #
        ###################
        routing = sample.get_routing_matrix()

        nodes = len(routing)
        # Remove diagonal from matrix
        paths = routing[~np.eye(routing.shape[0], dtype=bool)].reshape(routing.shape[0], -1)
        paths = paths.flatten()

        ###################
        #  EXTRACT LINKS  #
        ###################
        g = sample.get_topology_object()

        cap_mat = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=None)

        for node in range(g.number_of_nodes()):
            for adj in g[node]:
                cap_mat[node, adj] = g[node][adj][0]['bandwidth']

        print(cap_mat)

        links = np.where(np.ravel(cap_mat) != None)[0].tolist()

        link_capacities = (np.ravel(cap_mat)[links]).tolist()

        ids = list(range(len(links)))
        links_id = dict(zip(links, ids))

        path_ids = []
        for path in paths:
            new_path = []
            for i in range(0, len(path) - 1):
                src = path[i]
                dst = path[i + 1]
                new_path.append(links_id[src * nodes + dst])
            path_ids.append(new_path)

        ###################
        #   MAKE INDICES  #
        ###################
        link_indices = []
        path_indices = []
        sequ_indices = []
        segment = 0
        for p in path_ids:
            link_indices += p
            path_indices += len(p) * [segment]
            sequ_indices += list(range(len(p)))
            segment += 1

        traffic = sample.get_traffic_matrix()
        # Remove diagonal from matrix
        traffic = traffic[~np.eye(traffic.shape[0], dtype=bool)].reshape(traffic.shape[0], -1)

        result = sample.get_performance_matrix()
        # Remove diagonal from matrix
        result = result[~np.eye(result.shape[0], dtype=bool)].reshape(result.shape[0], -1)

        avg_bw = []
        pkts_gen = []
        delay = []
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                flow = traffic[i, j]['Flows'][0]
                avg_bw.append(flow['AvgBw'])
                pkts_gen.append(flow['PktsGen'])
                d = result[i, j]['AggInfo']['AvgDelay']
                delay.append(d)

        n_paths = len(path_ids)
        n_links = max(max(path_ids)) + 1

        yield {"bandwith": avg_bw, "packets": pkts_gen,
               "link_capacity": link_capacities,
               "links": link_indices,
               "paths": path_indices, "sequences": sequ_indices,
               "n_links": n_links, "n_paths": n_paths}, delay


def transformation(x, y):
    """Apply a transformation over all the samples included in the dataset.

        Args:
            x (dict): predictor variable.
            y (array): target variable.

        Returns:
            x,y: The modified predictor/target variables.
        """
    return x, y


def input_fn(data_dir, transform=True, repeat=True, shuffle=False):
    """This function uses the generator function in order to create a Tensorflow dataset

        Args:
            data_dir (string): Path of the data directory.
            transform (bool): If true, the data is transformed using the transformation function.
            repeat (bool): If true, the data is repeated. This means that, when all the data has been read,
                            the generator starts again.
            shuffle (bool): If true, the data is shuffled before being processed.

        Returns:
            tf.data.Dataset: Containing a tuple where the first value are the predictor variables and
                             the second one is the target variable.
        """
    ds = tf.data.Dataset.from_generator(lambda: generator(data_dir=data_dir, shuffle=shuffle),
                                        ({"bandwith": tf.float32, "packets": tf.float32,
                                          "link_capacity": tf.float32, "links": tf.int64,
                                          "paths": tf.int64, "sequences": tf.int64,
                                          "n_links": tf.int64, "n_paths": tf.int64},
                                        tf.float32),
                                        ({"bandwith": tf.TensorShape([None]), "packets": tf.TensorShape([None]),
                                          "link_capacity": tf.TensorShape([None]),
                                          "links": tf.TensorShape([None]),
                                          "paths": tf.TensorShape([None]),
                                          "sequences": tf.TensorShape([None]),
                                          "n_links": tf.TensorShape([]),
                                          "n_paths": tf.TensorShape([])},
                                         tf.TensorShape([None])))
    if transform:
        ds = ds.map(lambda x, y: transformation(x, y))

    if repeat:
        ds = ds.repeat()

    return ds
