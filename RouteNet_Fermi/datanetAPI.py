'''
 *
 * Copyright (C) 2020 Universitat Politècnica de Catalunya.
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

# -*- coding: utf-8 -*-

import os, tarfile, numpy, math, networkx, queue, random,traceback
from enum import IntEnum

import timeit

class DatanetException(Exception):
    """
    Exceptions generated when processing dataset
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg


class TimeDist(IntEnum):
    """
    Enumeration of the supported time distributions 
    """
    EXPONENTIAL_T = 0
    DETERMINISTIC_T = 1
    UNIFORM_T = 2
    NORMAL_T = 3
    ONOFF_T = 4
    PPBP_T = 5
    TRACE_T = 6
    EXTERNAL_PY_T = 7
    
    @staticmethod
    def getStrig(timeDist):
        if (timeDist == 0):
            return ("EXPONENTIAL_T")
        elif (timeDist == 1):
            return ("DETERMINISTIC_T")
        elif (timeDist == 2):
            return ("UNIFORM_T")
        elif (timeDist == 3):
            return ("NORMAL_T")
        elif (timeDist == 4):
            return ("ONOFF_T")
        elif (timeDist == 5):
            return ("PPBP_T")
        elif (timeDist == 6):
            return ("TRACE_T")
        elif (timeDist == 7):
            return ("EXTERNAL_PY_T")
        else:
            return ("UNKNOWN")

class SizeDist(IntEnum):
    """
    Enumeration of the supported size distributions 
    """
    DETERMINISTIC_S = 0
    UNIFORM_S = 1
    BINOMIAL_S = 2
    GENERIC_S = 3
    TRACE_S = 4
    
    @staticmethod
    def getStrig(sizeDist):
        if (sizeDist == 0):
            return ("DETERMINISTIC_S")
        elif (sizeDist == 1):
            return ("UNIFORM_S")
        elif (sizeDist == 2):
            return ("BINOMIAL_S")
        elif (sizeDist ==3):
            return ("GENERIC_S")
        elif (sizeDist ==4):
            return ("TRACE_S")
        else:
            return ("UNKNOWN")

class Sample:
    """
    Class used to contain the results of a single iteration in the dataset
    reading process.
    
    ...
    
    Attributes
    ----------
    global_packets : double
        Overall number of packets transmitteds in network
    global_losses : double
        Overall number of packets lost in network
    global_delay : double
        Overall delay in network
    maxAvgLambda: double
        This variable is used in our simulator to define the overall traffic 
        intensity  of the network scenario
    performance_matrix : NxN matrix
        Matrix where each cell [i,j] contains aggregated and flow-level
        information about transmission parameters between source i and
        destination j.
    traffic_matrix : NxN matrix
        Matrix where each cell [i,j] contains aggregated and flow-level
        information about size and time distributions between source i and
        destination j.
    routing_matrix : NxN matrix
        Matrix where each cell [i,j] contains the path, if it exists, between
        source i and destination j.
    topology_object : 
        Network topology using networkx format.
    port_stats: list-of-dict-of-dict data structure:
        The outer list contain a dict-of-dict for each node. The first dict contain
        the list of adjacents nodes and the last dict contain the parameters of the
        interface port.
    
    """
    
    global_packets = None
    global_losses = None
    global_delay = None
    maxAvgLambda = None
    
    performance_matrix = None
    traffic_matrix = None
    routing_matrix = None
    topology_object = None
    port_stats = None
    
    data_set_file = None
    _results_line = None
    _traffic_line = None
    _input_files_line = None
    _status_line = None
    _flowresults_line = None
    _link_usage_line = None
    _routing_file = None
    _graph_file = None
    
    def get_global_packets(self):
        """
        Return the number of packets transmitted in the network per time unit of this Sample instance.
        """
        
        return self.global_packets

    def get_global_losses(self):
        """
        Return the number of packets dropped in the network per time unit of this Sample instance.
        """
        
        return self.global_losses
    
    def get_global_delay(self):
        """
        Return the average per-packet delay over all the packets transmitted in the network in time units 
        of this sample instance.
        """
        
        return self.global_delay
    
    def get_maxAvgLambda(self):
        """
        Returns the maxAvgLamda used in the current iteration. This variable is used in our simulator to define 
        the overall traffic intensity of the network scenario.
        """
        
        return self.maxAvgLambda
        
    def get_performance_matrix(self):
        """
        Returns the performance_matrix of this Sample instance.
        """
        
        return self.performance_matrix
    
    def get_srcdst_performance(self, src, dst):
        """
        

        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary
            Information stored in the Result matrix for the requested src-dst.

        """
        return self.performance_matrix[src, dst]
        
    def get_traffic_matrix(self):
        """
        Returns the traffic_matrix of this Sample instance.
        """
        
        return self.traffic_matrix
    
    def get_srcdst_traffic(self, src, dst):
        """
        

        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary
            Information stored in the Traffic matrix for the requested src-dst.

        """
        
        return self.traffic_matrix[src, dst]
        
    def get_routing_matrix(self):
        """
        Returns the routing_matrix of this Sample instance.
        """
        
        return self.routing_matrix
    
    def get_srcdst_routing(self, src, dst):
        """
        

        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary
            Information stored in the Routing matrix for the requested src-dst.

        """
        return self.routing_matrix[src, dst]
        
    def get_topology_object(self):
        """
        Returns the topology in networkx format of this Sample instance.
        """
        
        return self.topology_object
    
    def get_network_size(self):
        """
        Returns the number of nodes of the topology.
        """
        return self.topology_object.number_of_nodes()
    
    def get_node_properties(self, id):
        """
        

        Parameters
        ----------
        id : int
            Node identifier.

        Returns
        -------
        Dictionary with the parameters of the node
        None if node doesn't exist

        """
        res = None
        
        if id in self.topology_object.nodes:
            res = self.topology_object.nodes[id] 
        
        return res
    
    def get_link_properties(self, src, dst):
        """
        

        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Dictionary with the parameters of the link
        None if no link exist between src and dst

        """
        res = None
        
        if dst in self.topology_object[src]:
            res = self.topology_object[src][dst][0] 
        
        return res
    
    def get_srcdst_link_bandwidth(self, src, dst):
        """
        

        Parameters
        ----------
        src : int
            Source node.
        dst : int
            Destination node.

        Returns
        -------
        Bandwidth in bits/time unit of the link between nodes src-dst or -1 if not connected

        """
        if dst in self.topology_object[src]:
            cap = float(self.topology_object[src][dst][0]['bandwidth'])
        else:
            cap = -1
            
        return cap
    
    def get_port_stats(self):
        """
        Returns the port_stats object of this Sample instance.
        """
        if (self.port_stats == None):
            raise DatanetException("ERROR: The processed dataset doesn't have port performance data")
        
        return self.port_stats
    
        
    def _set_data_set_file_name(self,file):
        """
        Sets the data set file from where the sample is extracted.
        """
        self.data_set_file = file
        
    def _set_performance_matrix(self, m):
        """
        Sets the performance_matrix of this Sample instance.
        """
        
        self.performance_matrix = m
        
    def _set_traffic_matrix(self, m):
        """
        Sets the traffic_matrix of this Sample instance.
        """
        
        self.traffic_matrix = m
        
    def _set_routing_matrix(self, m):
        """
        Sets the traffic_matrix of this Sample instance.
        """
        
        self.routing_matrix = m
        
    def _set_topology_object(self, G):
        """
        Sets the topology_object of this Sample instance.
        """
        
        self.topology_object = G
        
    def _set_global_packets(self, x):
        """
        Sets the global_packets of this Sample instance.
        """
        
        self.global_packets = x
        
    def _set_global_losses(self, x):
        """
        Sets the global_losses of this Sample instance.
        """
        
        self.global_losses = x
        
    def _set_global_delay(self, x):
        """
        Sets the global_delay of this Sample instance.
        """
        
        self.global_delay = x
        
    def _get_data_set_file_name(self):
        """
        Gets the data set file from where the sample is extracted.
        """
        return self.data_set_file
    
    def _get_path_for_srcdst(self, src, dst):
        """
        Returns the path between node src and node dst.
        """
        
        return self.routing_matrix[src, dst]
    
    def _get_timedis_for_srcdst (self, src, dst):
        """
        Returns the time distribution of traffic between node src and node dst.
        """
        
        return self.traffic_matrix[src, dst]['TimeDist']
    
    def _get_eqlambda_for_srcdst (self, src, dst):
        """
        Returns the equivalent lambda for the traffic between node src and node
        dst.
        """
        
        return self.traffic_matrix[src, dst]['EqLambda']
    
    def _get_timedistparams_for_srcdst (self, src, dst):
        """
        Returns the time distribution parameters for the traffic between node
        src and node dst.
        """
        
        return self.traffic_matrix[src, dst]['TimeDistParams']
    
    def _get_sizedist_for_srcdst (self, src, dst):
        """
        Returns the size distribution of traffic between node src and node dst.
        """
        
        return self.traffic_matrix[src, dst]['SizeDist']
    
    def _get_avgpktsize_for_srcdst_flow (self, src, dst):
        """
        Returns the average packet size for the traffic between node src and
        node dst.
        """
        
        return self.traffic_matrix[src, dst]['AvgPktSize']
    
    def _get_sizedistparams_for_srcdst (self, src, dst):
        """
        Returns the time distribution of traffic between node src and node dst.
        """
        
        return self.traffic_matrix[src, dst]['SizeDistParams']
    
    def _get_resultdict_for_srcdst (self, src, dst):
        """
        Returns the dictionary with all the information for the communication
        between node src and node dst regarding communication parameters.
        """
        
        return self.performance_matrix[src, dst]
    
    def _get_trafficdict_for_srcdst (self, src, dst):
        """
        Returns the dictionary with all the information for the communication
        between node src and node dst regarding size and time distribution
        parameters.
        """
        
        return self.traffic_matrix[src, dst]

class DatanetAPI:
    """
    Class containing all the functionalities to read the dataset line by line
    by means of an iteratos, and generate a Sample instance with the
    information gathered.
    """
    
    def __init__ (self, data_folder, intensity_values = [], topology_sizes = [],
                  shuffle=False, seed=None):
        """
        Initialization of the PasringTool instance

        Parameters
        ----------
        data_folder : str
            Folder where the dataset is stored.
        intensity_values : array of 1 or 2 integers
            User-defined intensity values used to constrain the reading process
            to the specified range.
        topology_sizes : array of integers
            User-defined topology sizes used to constrain the reading process
            to the specified values.
        shuffle: boolean
            Specify if all files should be shuffled. By default false
        Returns
        -------
        None.

        """
        
        self.data_folder = data_folder
        if (len(intensity_values) == 1):
            self.intensity_values = [intensity_values[0],intensity_values[0]]
        else:    
            self.intensity_values = intensity_values
        self.topology_sizes = topology_sizes
        self.shuffle = shuffle
        if seed:
            self.seed = seed
        else:
            self.seed = 1234

        self._all_tuple_files = []
        self._selected_tuple_files = []
        self._graphs_dic = {}
        self._routings_dic = {}
        self._external_param_dic = {"generate_autosimilar":["AR1-0","AR-a","samples"], "autosimilar_k2":["AR1-1","sigma","samples"]}
        for root, dirs, files in os.walk(self.data_folder):
            if ("graphs" not in dirs or "routings" not in dirs):
                continue
            files.sort()
            # Extend the list of files to process
            self._all_tuple_files.extend([(root, f) for f in files if f.endswith("tar.gz")])

    def get_available_files(self):
        """
        Get a list of all the dataset files located in the indicated data folder
        
        Returns
        -------
        Array of tuples where each tuple is (root directory, filename)
        
        """
        
        return (self._all_tuple_files.copy())
    
    def set_files_to_process(self, tuple_files_lst):
        """
        Set the list of files to be processed by the iterator. The files should belong to
        the list of tuples returned by get_available_files. 
        
        Parameters
        ----------
        tuple_files_lst: List of tuples
            List of tuples where each tuple is (path to file, filename)
        """
        if not type(tuple_files_lst) is list:
            raise DatanetException("ERROR: The argument of set_files_to_process should be a list of tuples -> [(root_dir,file),...]")
        for tuple in tuple_files_lst:
            if not type(tuple) is tuple or len(tuple) != 2:
                raise DatanetException("ERROR: The argument of set_files_to_process should be a list of tuples -> [(root_dir,file),...]")
            if (not tuple in self._all_tuple_files):
                raise DatanetException("ERROR: Selected tupla not belong to the list of tuples returned by get_available_files()")
        
        self._selected_tuple_files = tuple_files_lst.copy()

    def _readRoutingFile(self, routing_file, netSize):
        """
        Pending to compare against getSrcPortDst

        Parameters
        ----------
        routing_file : str
            File where the routing information is located.
        netSize : int
            Number of nodes in the network.

        Returns
        -------
        R : netSize x netSize matrix
            Matrix where each  [i,j] states what port node i should use to
            reach node j.

        """
        
        fd = open(routing_file,"r")
        R = numpy.zeros((netSize, netSize)) - 1
        src = 0
        for line in fd:
            camps = line.split(',')
            dst = 0
            for port in camps[:-1]:
                R[src][dst] = port
                dst += 1
            src += 1
        return (R)

    def _getRoutingSrcPortDst(self, G):
        """
        Return a dictionary of dictionaries with the format:
        node_port_dst[node][port] = next_node

        Parameters
        ----------
        G : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        is_multigraph = G.is_multigraph()
        
        node_port_dst = {}
        for node in G:
            port_dst = {}
            node_port_dst[node] = port_dst
            for destination in G[node].keys():
                if (is_multigraph):
                    port = G[node][destination][0]['port']
                else:
                    port = G[node][destination]['port']
                node_port_dst[node][port] = destination
        return(node_port_dst)
    
    def _create_routing_matrix_from_dst_routing_file(self, G, routing_file):
        """

        Parameters
        ----------
        G : graph
            Graph representing the network.
        routing_file : str
            File where the information about routing is located. The file is a 
            destination routing file.

        Returns
        -------
        MatrixPath : NxN Matrix
            Matrix where each cell [i,j] contains the path to go from node
            i to node j.

        """
        netSize = G.number_of_nodes()
        node_port_dst = self._getRoutingSrcPortDst(G)
        R = self._readRoutingFile(routing_file, netSize)
        MatrixPath = numpy.empty((netSize, netSize), dtype=object)
        for src in range (0,netSize):
            for dst in range (0,netSize):
                node = src
                path = [node]
                while (R[node][dst] != -1):
                    out_port = R[node][dst];
                    next_node = node_port_dst[node][out_port]
                    path.append(next_node)
                    node = next_node
                MatrixPath[src][dst] = path
        return (MatrixPath)
    
    def _create_routing_matrix_from_src_routing_dir(self, G, src_routing_dir):
        """

        Parameters
        ----------
        G : graph
            Graph representing the network.
        src_routing_dir : str
            Directory where we found the routing filesFile. One for each src node. 

        Returns
        -------
        MatrixPath : NxN Matrix
            Matrix where each cell [i,j] contains the path to go from node
            i to node j.

        """
        
        netSize = G.number_of_nodes()
        node_port_dst = self._getRoutingSrcPortDst(G)
        src_R = []
        for i in range(netSize):
            routing_file = os.path.join(src_routing_dir,"Routing_src_"+str(i)+".txt")
            src_R.append(self._readRoutingFile(routing_file, netSize))
        MatrixPath = numpy.empty((netSize, netSize), dtype=object)
        for src in range (0,netSize):
            R = src_R[src]
            for dst in range (0,netSize):
                node = src
                path = [node]
                while (R[node][dst] != -1):
                    out_port = R[node][dst];
                    next_node = node_port_dst[node][out_port]
                    path.append(next_node)
                    node = next_node
                MatrixPath[src][dst] = path
        return (MatrixPath)

    def _create_routing_matrix(self, G,routing_file):
        """

        Parameters
        ----------
        G : graph
            Graph representing the network.
        routing_file : str
            File where the information about routing is located.

        Returns
        -------
        MatrixPath : NxN Matrix
            Matrix where each cell [i,j] contains the path to go from node
            i to node j.

        """
        if (os.path.isfile(routing_file)):
            MatrixPath = self._create_routing_matrix_from_dst_routing_file(G,routing_file)
        elif(os.path.isdir(routing_file)):
            MatrixPath = self._create_routing_matrix_from_src_routing_dir(G,routing_file)
        
        return (MatrixPath)

    def _generate_graphs_dic(self, path):
        """
        Return a dictionary with networkx objects generated from the GML
        files found in path
 
        Parameters
        ----------
        path : str
            Direcotory where the graphs files are located.
 
        Returns
        -------
        Returns a dictionary where keys are the names of GML files found in path 
        and the values are the networkx object generated from the GML files.
         
        """
        
        graphs_dic = {}
        for topology_file in os.listdir(path):
            G = networkx.read_gml(path+"/"+topology_file, destringizer=int)
            graphs_dic[topology_file] = G
        
        return graphs_dic
    
    def _graph_links_update(self,G,file):
        """
        Updates the graph with the link information of the file
        
        Parameters
        ----------
        G : graph
            Graph object to be updated
        file: str
            file name that contains the information of the links to be modified: src;dst;bw (bps)
            
        Returns
        -------
        None
        
        """
        
        try:
            fd = open(file,"r")
        except:
            print ("ERROR: %s not exists" % (file))
            exit(-1)
        
        for line in fd:
            aux = line.split(";")
            G[int(aux[0])][int(aux[1])][0]["bandwidth"] = int(aux[2])

    def __iter__(self):
        """
        

        Yields
        ------
        s : Sample
            Sample instance containing information about the last line read
            from the dataset.

        """
        
        g = None
        
        if (len(self._selected_tuple_files) > 0):
            tuple_files = self._selected_tuple_files
        else:
            tuple_files = self._all_tuple_files

        if self.shuffle:
            random.Random(1234).shuffle(tuple_files)
        ctr = 0
        for root, file in tuple_files:
            try:
                it = 0 
                tar = tarfile.open(os.path.join(root, file), 'r:gz')
                dir_info = tar.next().name
                if (dir_info[-1]=="/"):
                    dir_info = dir_info[:-1]
                
                results_file = tar.extractfile(dir_info + "/simulationResults.txt")
                traffic_file = tar.extractfile(dir_info + "/traffic.txt")
                status_file = tar.extractfile(dir_info + "/stability.txt")
                input_files = tar.extractfile(dir_info + "/input_files.txt")
                if (dir_info + "/flowSimulationResults.txt" in tar.getnames()):
                    flowresults_file = tar.extractfile(dir_info + "/flowSimulationResults.txt")
                else:
                    flowresults_file = None
                if (dir_info + "/linkUsage.txt" in tar.getnames()):
                    link_usage_file = tar.extractfile(dir_info + "/linkUsage.txt")
                else:
                    link_usage_file = None
                while(True):
                    s = Sample()
                    s._set_data_set_file_name(os.path.join(root, file))
                    
                    s._results_line = results_file.readline().decode()[:-2]
                    s._traffic_line = traffic_file.readline().decode()[:-1]
                    if (flowresults_file):
                        s._flowresults_line = flowresults_file.readline().decode()[:-2]
                    s._status_line = status_file.readline().decode()[:-1]
                    s._input_files_line = input_files.readline().decode()[:-1]
                    if (link_usage_file):
                        s._link_usage_line = link_usage_file.readline().decode()[:-1]
                    if (len(s._results_line) == 0) or (len(s._traffic_line) == 0):
                        break
                    if (not ";OK;" in s._status_line):
                        print ("Removed iteration: "+s._status_line)
                        continue;
                    
                    # Filter traffic intensity
                    if (len(self.intensity_values) != 0):
                        ptr = s._traffic_line.find('|')
                        specific_intensity = float(s._traffic_line[0:ptr])
                        if(specific_intensity < self.intensity_values[0]) or (specific_intensity > self.intensity_values[1]):
                            continue
                    
                    used_files = s._input_files_line.split(';')
                    s._graph_file = os.path.join(root,"graphs",used_files[1])
                    s._routing_file = os.path.join(root,"routings",used_files[2])
                    if (s._graph_file in self._graphs_dic):
                        g = self._graphs_dic[s._graph_file]
                    else:
                        g = networkx.read_gml(s._graph_file, destringizer=int)
                        self._graphs_dic[s._graph_file] = g
                    
                    # Filter topology size
                    if (len(self.topology_sizes) != 0 and not len(g) in self.topology_sizes):
                        continue
                
                    if (len(used_files) == 4):
                        self._graph_links_update(g,os.path.join(root,"links_bw",used_files[3]))
                    
                    # XXX We considerer that all graphs using the same routing file have the same topology
                    if (s._routing_file in self._routings_dic):
                        routing_matrix = self._routings_dic[s._routing_file]
                    else:
                        routing_matrix = self._create_routing_matrix(g,s._routing_file)
                        self._routings_dic[s._routing_file] = routing_matrix
                    
                    s._set_routing_matrix(routing_matrix)
                    s._set_topology_object(g)
                    self._process_flow_results(s)
                    if (s._link_usage_line):
                        self._process_link_usage(s)
                    it +=1
                    yield s
            except (GeneratorExit,SystemExit) as e:
                raise
            except:
                traceback.print_exc()
                print ("Error in the file: %s   iteration: %d" % (file,it))
                exit(1)
                    
            ctr += 1
            #print("Progress check: %d/%d" % (ctr,len(tuple_files)))
    
    def _process_flow_results(self, s):
        """
        

        Parameters
        ----------
        s : Sample
            Instance of Sample associated with the current iteration.

        Returns
        -------
        None.

        """
        
        q_flows = queue.Queue()
        first_params = s._results_line.split('|')[0].split(',')
        first_params = list(map(float, first_params))
        s._set_global_packets(first_params[0])
        s._set_global_losses(first_params[1])
        s._set_global_delay(first_params[2])
        r = s._results_line[s._results_line.find('|')+1:].split(';')
        if (s._flowresults_line):
            f = s._flowresults_line.split(';')
        else:
            f = r
        
        ptr = s._traffic_line.find('|')
        t = s._traffic_line[ptr+1:].split(';')
        s.maxAvgLambda = float(s._traffic_line[:ptr])
        sim_time  = float(s._status_line.split(';')[0])
        
        m_result = []
        m_traffic = []
        for i in range(0,len(r), int(math.sqrt(len(r)))):
            new_result_row = []
            new_traffic_row = []
            for j in range(i, i+int(math.sqrt(len(r)))):
                dict_result_srcdst = {}
                aux_agg_ = r[j].split(',')
                aux_agg = list(map(float, aux_agg_))
                dict_result_agg = {'PktsDrop':aux_agg[2], "AvgDelay":aux_agg[3], "AvgLnDelay":aux_agg[4], "p10":aux_agg[5], "p20":aux_agg[6], "p50":aux_agg[7], "p80":aux_agg[8], "p90":aux_agg[9], "Jitter":aux_agg[10]}
                
                lst_result_flows = []
                aux_result_flows = f[j].split(':')
                for flow in aux_result_flows:
                    dict_result_tmp = {}
                    tmp_result_flow = flow.split(',')
                    tmp_result_flow = list(map(float, tmp_result_flow))
                    q_flows.put([tmp_result_flow[0], tmp_result_flow[1]])
                    dict_result_tmp = {'PktsDrop':tmp_result_flow[2], "AvgDelay":tmp_result_flow[3], "AvgLnDelay":tmp_result_flow[4], "p10":tmp_result_flow[5], "p20":tmp_result_flow[6], "p50":tmp_result_flow[7], "p80":tmp_result_flow[8], "p90":tmp_result_flow[9], "Jitter":tmp_result_flow[10]}
                    lst_result_flows.append(dict_result_tmp)
                
                dict_traffic_srcdst = {}
                # From kbps to bps
                dict_traffic_agg = {'AvgBw':aux_agg[0]*1000,
                                    'PktsGen':aux_agg[1],
                                    'TotalPktsGen':aux_agg[1]*sim_time}
                lst_traffic_flows = []
                aux_traffic_flows = t[j].split(':')
                for flow in aux_traffic_flows:
                    dict_traffic = {}
                    q_values_for_flow = q_flows.get()
                    tmp_traffic_flow = flow.split(',')
                    offset = self._timedistparams(tmp_traffic_flow,dict_traffic)
                    if offset != -1:
                        self._sizedistparams(tmp_traffic_flow, offset, dict_traffic)
                        # From kbps to bps
                        dict_traffic['AvgBw'] = q_values_for_flow[0]*1000
                        dict_traffic['PktsGen'] = q_values_for_flow[1]
                        dict_traffic['TotalPktsGen'] = sim_time * dict_traffic['PktsGen']
                        dict_traffic['ToS'] = int(tmp_traffic_flow[-1])
                    if (len(dict_traffic.keys())!=0):
                        lst_traffic_flows.append (dict_traffic)
                
                dict_result_srcdst['AggInfo'] = dict_result_agg
                dict_result_srcdst['Flows'] = lst_result_flows
                dict_traffic_srcdst['AggInfo'] = dict_traffic_agg
                dict_traffic_srcdst['Flows'] = lst_traffic_flows
                new_result_row.append(dict_result_srcdst)
                new_traffic_row.append(dict_traffic_srcdst)
                
            m_result.append(new_result_row)
            m_traffic.append(new_traffic_row)
        m_result = numpy.asmatrix(m_result)
        m_traffic = numpy.asmatrix(m_traffic)
        s._set_performance_matrix(m_result)
        s._set_traffic_matrix(m_traffic)

    def _timedistparams(self, data, dict_traffic):
        """
        

        Parameters
        ----------
        data : List
            List of all the flow traffic parameters to be processed.
        dict_traffic: dictionary
            Dictionary to fill with the time distribution information
            extracted from data

        Returns
        -------
        offset : int
            Number of elements read from the list of parameters data

        """
        
        if data[0] == "0": 
            dict_traffic['TimeDist'] = TimeDist.EXPONENTIAL_T
            params = {}
            params['EqLambda'] = float(data[1])
            params['AvgPktsLambda'] = float(data[2])
            params['ExpMaxFactor'] = float(data[3])
            dict_traffic['TimeDistParams'] = params
            return 4
        elif data[0] == "1":
            dict_traffic['TimeDist'] = TimeDist.DETERMINISTIC_T
            params = {}
            params['EqLambda'] = float(data[1])
            params['AvgPktsLambda'] = float(data[2])
            dict_traffic['TimeDistParams'] = params
            return 3
        elif data[0] == "2":
            dict_traffic['TimeDist'] = TimeDist.UNIFORM_T
            params = {}
            params['EqLambda'] = float(data[1])
            params['MinPktLambda'] = float(data[2])
            params['MaxPktLambda'] = float(data[3])
            dict_traffic['TimeDistParams'] = params
            return 4
        elif data[0] == "3":
            dict_traffic['TimeDist'] = TimeDist.NORMAL_T
            params = {}
            params['EqLambda'] = float(data[1])
            params['AvgPktsLambda'] = float(data[2])
            params['StdDev'] = float(data[3])
            dict_traffic['TimeDistParams'] = params
            return 4
        elif data[0] == "4":
            dict_traffic['TimeDist'] = TimeDist.ONOFF_T
            params = {}
            params['EqLambda'] = float(data[1])
            params['PktsLambdaOn'] = float(data[2])
            params['AvgTOff'] = float(data[3])
            params['AvgTOn'] = float(data[4])
            params['ExpMaxFactor'] = float(data[5])
            dict_traffic['TimeDistParams'] = params
            return 6
        elif data[0] == "5":
            dict_traffic['TimeDist'] = TimeDist.PPBP_T
            params = {}
            params['EqLambda'] = float(data[1])
            params['BurstGenLambda'] = float(data[2])
            params['Bitrate'] = float(data[3])
            params['ParetoMinSize'] = float(data[4])
            params['ParetoMaxSize'] = float(data[5])
            params['ParetoAlfa'] = float(data[6])
            params['ExpMaxFactor'] = float(data[7])
            dict_traffic['TimeDistParams'] = params
            return 8
        elif data[0] == "6":
            dict_traffic['TimeDist'] = TimeDist.TRACE_T
            params = {}
            params['EqLambda'] = float(data[1])
            dict_traffic['TimeDistParams'] = params
            return 2
        elif data[0] == "7":
            dict_traffic['TimeDist'] = TimeDist.EXTERNAL_PY_T
            try:
                params_list = self._external_param_dic[data[2]]
            except:
                print ("Error: No external file descriptor for "+data[2])
                return -1
            params = {}
            params['EqLambda'] = float(data[1])
            params['Distribution'] = params_list[0]
            pos = 3 # 0 is EqLambda and 1 is name of the used module
            for pname in params_list[1:]:
                params[pname] = float(data[pos])
                pos += 1
            dict_traffic['TimeDistParams'] = params
            return pos
        else: return -1
    
    def _sizedistparams(self, data, starting_point, dict_traffic):
        """
        

        Parameters
        ----------
        data : List
            List of all the flow traffic parameters to be processed.
        starting_point : int
            Point of the overall traffic file line where the extraction of
            data regarding the size distribution should start.
        dict_traffic : dictionary
            Dictionary to fill with the size distribution information
            extracted from data

        Returns
        -------
        ret : int
            0 if it finish successfully and -1 otherwise

        """
        
        if data[starting_point] == "0":
            dict_traffic['SizeDist'] = SizeDist.DETERMINISTIC_S
            params = {}
            params['AvgPktSize'] = float(data[starting_point+1])
            dict_traffic['SizeDistParams'] = params
        elif data[starting_point] == "1":
            dict_traffic['SizeDist'] = SizeDist.UNIFORM_S
            params = {}
            params['AvgPktSize'] = float(data[starting_point+1])
            params['MinSize'] = float(data[starting_point+2])
            params['MaxSize'] = float(data[starting_point+3])
            dict_traffic['SizeDistParams'] = params
        elif data[starting_point] == "2":
            dict_traffic['SizeDist'] = SizeDist.BINOMIAL_S
            params = {}
            params['AvgPktSize'] = float(data[starting_point+1])
            params['PktSize1'] = float(data[starting_point+2])
            params['PktSize2'] = float(data[starting_point+3])
            dict_traffic['SizeDistParams'] = params
        elif data[starting_point] == "3":
            dict_traffic['SizeDist'] = SizeDist.GENERIC_S
            params = {}
            params['AvgPktSize'] = float(data[starting_point+1])
            params['NumCandidates'] = float(data[starting_point+2])
            for i in range(0, int(data[starting_point+2]) * 2, 2):
                params["Size_%d"%(i/2)] = float(data[starting_point+3+i])
                params["Prob_%d"%(i/2)] = float(data[starting_point+4+i])
            dict_traffic['SizeDistParams'] = params
        elif data[starting_point] == 4:
            dict_traffic['SizeDist'] = SizeDist.TRACE_S
            params = {}
            try:
                params['AvgPktSize'] = round(dict_traffic['AvgBw']/dict_traffic['PktsGen'],6)
            except:
                params['AvgPktSize'] = 0
            dict_traffic['SizeDistParams'] = params
        else:
            return -1
        return 0

    def _process_link_usage(self,s):
        """

        Parameters
        ----------
        s : Sample
            Instance of Sample associated with the current iteration.

        Returns
        -------
        None.

        """
        # port_state is an array of the nodes containing a dictionary with the adjacent nodes. 
        # Each adjacent node contains a dictionary with performance metrics
        port_stat = []
        l = s._link_usage_line.split(";")
        netSize = s.get_network_size()
        for i in range(netSize):
            port_stat.append({})
            for j in range(netSize):
                params_lst = l[i*netSize+j].split(":")
                params = params_lst[0].split(",")
                if (params[0] == "-1"):
                    continue
                link_stat = {}
                link_stat["utilization"] = float(params[0])
                link_stat["losses"] = float(params[1])
                link_stat["avgPacketSize"] = float(params[2])
                qos_queue_stat_lst = []
                for params in params_lst[1:]:
                    queue_params = params.split(",")
                    qos_queue_stat = {"utilization":float(queue_params[0]),
                                  "losses":float(queue_params[1]),
                                  "avgPortOccupancy":float(queue_params[2]),
                                  "maxQueueOccupancy":float(queue_params[3]),
                                  "avgPacketSize":float(queue_params[4])}
                    qos_queue_stat_lst.append(qos_queue_stat)
                link_stat["qosQueuesStats"] = qos_queue_stat_lst;
                port_stat[i][j] = link_stat
#         
        s.port_stats = port_stat
