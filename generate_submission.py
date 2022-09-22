#!/usr/bin/python3

import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append("RouteNet_Fermi")
from model import RouteNet_Fermi
import zipfile
import math
import traceback
import datanetAPI
import networkx as nx
import numpy as np
import shutil
import time
import tensorflow as tf



##########################
### Dataset validation ###
##########################


max_topology_size = 10
min_link_bandwidth = 10000
max_link_bandwidth = 400000
min_buffer_size = 8000
max_buffer_size = 64000
min_avg_bw = 10
max_avg_bw = 10000
min_pkt_size = 250
max_pkt_size = 2000

class SimWorkerException(Exception):
    """
    Exceptions generated when processing simulation file
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

def get_root_path(path_to_extract_to):
    files = os.listdir(path_to_extract_to)
    if ("dataset" in files and "model" in files):
        return (path_to_extract_to)
    if (len(files)!=1):
        return (None)
    return (get_root_path(os.path.join(path_to_extract_to,files[0])))


def check_node_config(node):
    if (not "schedulingPolicy" in node):
        raise SimWorkerException("Nodes should specify schedulingPolicy. It could be 'FIFO', 'SP', 'WFQ' or 'DRR'.")
    
    policy = node["schedulingPolicy"]
    if (policy != "FIFO" and policy != "SP" and policy != "WFQ" and policy != "DRR"):
        raise SimWorkerException("Node schedulingPolicy could be 'FIFO', 'SP', 'WFQ' or 'DRR'. ({})".format(policy))
    
    if (policy == "WFQ" or policy == "DRR"):
        if (not "schedulingWeights" in node):
           raise SimWorkerException("When a node is configured with 'WFQ' or 'DRR', schedulingWeights should be specified.".format(policy))
        
        weights_str = node["schedulingWeights"]
        try:
            weights_lst = list(map(float,weights_str.split(",")))
        except:
            raise SimWorkerException("Error converting the weights of the scheduling policy. ({})".format(weights_str))
        
        if (len(weights_lst) != 3):
            raise SimWorkerException("Three weights should be specified for {} policy".format(policy))
        
        if (not math.isclose (np.sum(weights_lst), 100)):
            raise SimWorkerException("The weights for {} policy should sum 100".format(policy))

    
            
    
    if (not "bufferSizes" in node):
        raise SimWorkerException("Nodes should specify buffer size of queues")
    
    buffer_size_str = node["bufferSizes"]
    if (type(buffer_size_str) is str):
        buffer_size_lst = buffer_size_str.split(",")
        buffer_size_lst = list(map(int,buffer_size_lst))
    else:
        buffer_size_lst = [buffer_size_str]
   
    if (len (set(buffer_size_lst)) != 1):
        raise SimWorkerException("The buffer size should be tha same for all queues of a node")
    
    if (buffer_size_lst[0] < min_buffer_size or buffer_size_lst[0] > max_buffer_size):
        raise SimWorkerException("Buffer size of queues should be between {} and {} ({})".format(min_buffer_size,max_buffer_size,buffer_size_lst[0]))    
    
    return (True)

def check_edge_config(edge):
    if(not "bandwidth" in edge):
        raise SimWorkerException("Edges of topology file should specify bandwidth")
    try:
        bw = int(edge["bandwidth"])
    except:
        raise SimWorkerException("Topology: The bandwidth of edges should be an integer ({})".format(edge["bandwidth"]))
    if (bw % 1000 != 0):
        raise SimWorkerException("Topology: The bandwidth of edges should be multiple of 1000 ({})".format(edge["bandwidth"]))
    if (bw > max_link_bandwidth or bw < min_link_bandwidth):
        raise SimWorkerException("Topology: The bandwidth of edges should be between {} and {} ({})".format(min_link_bandwidth,max_link_bandwidth,edge["bandwidth"]))
    
    return (True)



def sample_validate_topology(sample):
    G = sample.get_topology_object()
    if (len(G) > max_topology_size):
        raise SimWorkerException("Topology size is {}. The maximum valid size is {}.".format(len(G),max_topology_size))

    
    if (not nx.is_strongly_connected(G)):
        raise SimWorkerException("Topology is not strongly connected.")
    
    for i in range(len(G)):
        try:
            check_node_config(G.nodes[i])
        except SimWorkerException:
            raise
    
    for e in G.edges():
        try:
            check_edge_config(G[e[0]][e[1]])
        except SimWorkerException:
            raise
    
    if (G.graph["levelsToS"] != 3):
        raise SimWorkerException("Not valid levels of ToS value.")

#################################################################


def sample_validate_tm(sample):
    tm = sample.get_traffic_matrix()
    net_size = sample.get_network_size()


    for i in range(net_size):
        for j in range(net_size):
            if (i==j):
                continue
            avg_bw = tm[i,j]["Flows"][0]["TimeDistParams"]["EqLambda"]
            if (avg_bw < min_avg_bw or avg_bw > max_avg_bw):
                raise SimWorkerException("Average bandwidth of the path should be between {} and {}: {} ".format(min_avg_bw,max_avg_bw,avg_bw))
            
            if (len(tm[i,j]["Flows"]) > 1):
                raise SimWorkerException("Only one flow per path is allowed")
            
            path_traffic = tm[i,j]["Flows"][0]

            # Check pkt size parameters
            if (path_traffic["SizeDist"] != datanetAPI.SizeDist.GENERIC_S):
                raise SimWorkerException("Invalid size distribution")
            
            pkt_size_param = path_traffic["SizeDistParams"]
            num_candidates = int(pkt_size_param["NumCandidates"])

            if (num_candidates > 5):
                raise SimWorkerException("A maximum of 5 packet size can be defined in a packet size distribution.")

            all_pkt_size_prob = 0
            for cand in range(num_candidates):
                pkt_size = pkt_size_param["Size_{}".format(cand)]
                all_pkt_size_prob += pkt_size_param["Prob_{}".format(cand)]
                if (pkt_size < min_pkt_size or pkt_size > max_pkt_size):
                    raise SimWorkerException("Packet Size should be between {} and {}".format(min_pkt_size,max_pkt_size))

            if (not math.isclose(all_pkt_size_prob,1)):
                raise SimWorkerException("The sum of probabilities of all packets sizes should be one")
            
            # Check time dist parameters
            time_dist = path_traffic["TimeDist"]
            if (time_dist != datanetAPI.TimeDist.EXPONENTIAL_T and time_dist != datanetAPI.TimeDist.DETERMINISTIC_T and time_dist != datanetAPI.TimeDist.ONOFF_T):
                raise SimWorkerException("Invalid time distribution")  

#################################################################

def validate_dataset(dataset_path):
    if (not os.path.isdir(dataset_path)):
        print ("ERROR: The dataset folder doesn't exist.")
        return (1)

    print("Checking the dataset...")
    files_lst = os.listdir(dataset_path)

    if (not "graphs" in files_lst or not "routings" in files_lst):
        print ("ERROR: The selected folder doesn't contain a dataset or, it is in a subfolder.")
        return (1)

    
    

    sample_num = 0
    APE = []
    
    reader = datanetAPI.DatanetAPI(dataset_path)
    it = iter(reader)
    num_samples = 0
    sample_pos = 0
    
    has_errors = False
    for sample in it:
        sample_pos += 1
        if (sample == None):
            print("{}: datanetAPI error processing sample.\n".format(sample_pos - 1))
            has_errors = True
            continue
        try:
            sample_validate_topology(sample)
            sample_validate_tm(sample)
        except SimWorkerException as e:
            print("{}:{}\n".format(sample_pos - 1,e))
            has_errors = True
            continue
        num_samples += 1
    
    if (has_errors):
        print("==> Invalid dataset. Some samples have errors.")
        return (1)
    elif (num_samples > 100):
        print("==> Invalid dataset. The dataset contains more than 100 samples")
        return (1)
    else:
        print("==> Dataset validated. Number of samples: {}".format(num_samples))
        return (0)

#################################################################

def validate_checkpoint(checkpoint):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    model = RouteNet_Fermi()

    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(loss=loss_object,
                optimizer=optimizer,
                run_eagerly=False)

    try:
        model.load_weights(checkpoint).expect_partial()
    except Exception as e:
        traceback.print_exc()
        return(1)

    return (0)

#################################################################

def cp_dataset(src_path,dst_path):
    for root, dirs, files in os.walk(src_path):
        for file in files:
            path_file = os.path.join(root,file)
            shutil.copy2(path_file,dst_path) 

def zipfolder(foldername, target_dir):            
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])

def generate_submission(dataset_path,checkpoint_path,checkpoint,out_dir):
    dst_path_dataset = os.path.join(out_dir,"dataset")
    dst_path_model = os.path.join(out_dir,"model")
    os.mkdir(out_dir)
    os.mkdir(dst_path_model)
    shutil.copytree(dataset_path,dst_path_dataset)
    files_lst = [os.path.join(checkpoint_path,f) for f in os.listdir(checkpoint_path) if f.startswith(checkpoint)]
    if (len(files_lst) != 2):
        print ("ERROR: More than 2 file with the same chakpoint: ",files_lst)
        exit(1)
    shutil.copy2(files_lst[0],dst_path_model)
    shutil.copy2(files_lst[1],dst_path_model)

    zipfolder(out_dir, out_dir)
    shutil.rmtree(out_dir)


############################  MAIN ##############################
if __name__ == "__main__":
    dataset_path = input("Indicate the path to the folder of the dataset used to train the model: ")

    if (validate_dataset(dataset_path) != 0):
        exit(1)

    print ("==> Dataset validated\n\n")

    while (True):
        checkpoint_path = input("Indicate the path to the folder of the checkpoints used to train the model: ")
        if (not os.path.isdir(checkpoint_path)):
            print ("ERROR: Checkpoints folder doesn't exist")
            time.sleep(2)
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        checkpoints_lst = [f[:-6] for f in os.listdir(checkpoint_path) if f.endswith(".index")]
        if (len(checkpoints_lst) == 0):
            print ("ERROR: No checkpoints found in the selected path")
            time.sleep(2)
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        break
    while (True):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("List of the checkpoints found: ")
        print(checkpoints_lst)
        checkpoint = input("Select one of the checkpoints. It is not required to be the last one: ")
        if (not checkpoint in checkpoints_lst):
            print("The checkpoint {} doesn't exist.".format(checkpoint))
            time.sleep(2)
            continue
        break


    if (validate_checkpoint(os.path.join(checkpoint_path,checkpoint)) != 0):
        print ("ERROR: Invalid checkpoint")
        exit(1)


    print ("==> Checkpoint validated\n\n")

    out_path = input ("Select destination path where to generate the submission file, or leave empty to use the root instead: ")
    if (out_path !=""):
        if(not os.path.isdir(out_path)):
            res = input("Folder doesn't exist. Do you want to create it? (y/n) ")
            if (res=='y' or res=='Y'):
                os.makedirs(out_path)
            else:
                print ("Canceled process...")
                exit(1)

    name = input ("Submission name :")


    dst_folder = os.path.join(out_path,name)

    generate_submission(dataset_path,checkpoint_path,checkpoint,dst_folder)
    print ("==> ZIP submission file generated\n\n")



