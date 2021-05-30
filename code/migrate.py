'''
 *
 * Copyright (C) 2021 Universitat Politècnica de Catalunya.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
'''

import numpy as np
import random
import networkx as nx
from networkx.readwrite import json_graph
import json
import tarfile
import os
from datanetAPI import DatanetAPI
from joblib import Parallel, delayed
import tempfile
import shutil
from shutil import copyfile, copytree
import multiprocessing

DATA_DIR = '../data/sample_data/train'
DESTINATION_DIR = '../data/train'
NUM_SAMPLES = None
JSON_MAX_SAMPLES = 10
NUM_CORES = multiprocessing.cpu_count() - 2


def network_to_hypergraph(network_graph, routing_matrix, traffic_matrix, performance_matrix, port_stats):
    G = nx.DiGraph(network_graph)
    R = routing_matrix
    T = traffic_matrix
    D = performance_matrix
    P = port_stats

    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:

                if G.has_edge(src, dst):
                    D_G.add_node('l_{}_{}'.format(src, dst),
                                 entity='link',
                                 capacity=G.edges[src, dst]['bandwidth'],
                                 occupancy=P[src][dst]['qosQueuesState'][0]['avgPortOccupancy'] /
                                           G.nodes[src]['queueSizes'])

                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                        D_G.add_node('p_{}_{}_{}'.format(src, dst, f_id),
                                     entity='path',
                                     traffic=T[src, dst]['Flows'][f_id]['AvgBw'],
                                     packets=T[src, dst]['Flows'][f_id]['PktsGen'],
                                     delay=D[src, dst]['Flows'][f_id]['AvgDelay'])

                        for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                            D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id), 'l_{}_{}'.format(h_1, h_2))
                            D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'p_{}_{}_{}'.format(src, dst, f_id))

    D_G.remove_nodes_from([node for node, out_degree in D_G.out_degree() if out_degree == 0])

    return D_G


def generator(data_dir):
    tool = DatanetAPI(data_dir)
    it = iter(tool)
    for sample in it:
        G_copy = sample.get_topology_object().copy()
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        P = sample.get_performance_matrix()
        L = sample.get_port_stats()
        HG = network_to_hypergraph(network_graph=G_copy,
                                   routing_matrix=R,
                                   traffic_matrix=T,
                                   performance_matrix=P,
                                   port_stats=L)
        yield HG


def save_dataset(temp_dir, path):
    temp_dir = os.path.normpath(temp_dir)
    gen = generator(temp_dir)
    samples = []
    proc_samples = 0
    num_json = 0
    for graph in gen:
        samples.append(json_graph.node_link_data(graph))
        proc_samples += 1
        if proc_samples % JSON_MAX_SAMPLES == 0:
            save_dir = os.path.join(DESTINATION_DIR,
                                    os.path.basename(path + '_{}_{}.json'.format(os.path.basename(temp_dir), num_json)))
            num_json += 1
            with open(save_dir, "w") as f:
                json.dump(samples, f)
            samples = []
        if proc_samples == NUM_SAMPLES:
            break

    if len(samples) > 0:
        save_dir = os.path.join(DESTINATION_DIR,
                                os.path.basename(path + '_{}_{}.json'.format(os.path.basename(temp_dir), num_json)))
        num_json += 1
        with open(save_dir, "w") as f:
            json.dump(samples, f)


def divide_directory(path, num_cores):
    data_files = [f for f in os.listdir(path) if f.endswith('.tar.gz')]
    random.shuffle(data_files)
    splited_files = np.array_split(np.array(data_files), num_cores)
    dir_count = 0
    gen_dir = []
    dirpath = tempfile.mkdtemp()

    for dir in splited_files:
        temp_dir = dirpath + '/' + str(dir_count) + '/'
        os.mkdir(temp_dir)
        gen_dir.append(temp_dir)
        copytree(os.path.join(path, "graphs"), os.path.join(temp_dir, 'graphs'))
        copytree(os.path.join(path, 'routings'), os.path.join(temp_dir, 'routings'))
        for f in dir:
            copyfile(path + '/' + f, temp_dir + f)
        dir_count += 1

    return dirpath, gen_dir


if not os.path.exists(DESTINATION_DIR):
    os.makedirs(DESTINATION_DIR)

subdirs = [x[0] for x in os.walk(DATA_DIR) if
           os.path.exists(os.path.join(x[0], 'graphs')) and os.path.exists(os.path.join(x[0], 'routings'))]

for dir in subdirs:
    print("STARTING PROCESSING: {}".format(dir))
    num_cores = min(NUM_CORES, len([f for f in os.listdir(dir)
                                    if f.endswith('.tar.gz') and os.path.isfile(
            os.path.join(dir, f))]))

    if num_cores == 0:
        continue

    temp_dir, gen_dir = divide_directory(dir, num_cores)

    Parallel(n_jobs=num_cores)(delayed(save_dataset)(g_dir, dir) for g_dir in gen_dir)

    shutil.rmtree(temp_dir)
