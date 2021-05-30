# RouteNet - Graph Neural Networking challenge 2020

#### Organized as part of "ITU AI/ML in 5G challenge"

#### Challenge website: https://bnn.upc.edu/challenge2020

#### Contact mailing list: challenge-kdn@mail.knowledgedefinednetworking.org
(Please, note that you first need to subscribe at: https://mail.knowledgedefinednetworking.org/cgi-bin/mailman/listinfo/challenge-kdn).

[RouteNet](https://arxiv.org/abs/1901.08113) is a Graph Neural Network (GNN) model that estimates per-source-destination performance metrics (e.g., delay, jitter, loss) in networks. Thanks to its GNN architecture that operates over graph-structured data, RouteNet revealed an unprecedented ability to learn and model the complex relationships among topology, routing and input traffic in networks. As a result, it was able to make performance predictions with similar accuracy than costly packet-level simulators even in network scenarios unseen during training. This provides network operators with a functional tool able to make accurate predictions of end-to-end Key Performance Indicators (e.g., delay, jitter, loss).

<p align="center"> 
  <img src="/assets/routenet_scheme.png" width="600" alt>
</p>

## Quick Start
### Requirements
We strongly recommend use Python 3.7 or greater, since lower versions of Python may cause problems to define all the PATH environment variables.
The following packages are required:
* IGNNITION == 1.0.1. You can install it following the official [IGNNITION installation guide.](https://ignnition.net/doc/installation/)

IGNNITION is a TensorFlow-based framework for fast prototyping of GNNs. It provides a codeless programming interface, where users can implement their own GNN models in a YAML file, without writing a single line of TensorFlow. With this tool, network engineers are able to create their own GNN models in a matter of few hours. IGNNITION also incorporates a set of tools and functionalities that guide users during the design and implementation process of the GNN.
### Testing the installation
In order to test the installation, we provide a toy dataset that contains some random samples. You can simply verify your installation by executing the [main.py](/code/main.py) file.
```
python main.py
```

You should see something like this:
```
Processing the described model...
---------------------------------------------------------------------------


Creating the GNN model...
---------------------------------------------------------------------------


Generating the computational graph... 
---------------------------------------------------------------------------


Starting the training and validation process...
---------------------------------------------------------------------------

Number of devices: 1
Epoch 1/5
1000/1000 [==============================] - 125s 125ms/step - loss: 1.2603 - mean_absolute_percentage_error: 59.3345 - denorm_mean_absolute_percentage_error: 68.9152 - val_loss: 3.0187 - val_mean_absolute_percentage_error: 258.9615 - val_denorm_mean_absolute_percentage_error: 259.1504

```

### Training, evaluation and prediction
IGNNITION provides two methods, one for training and evaluation and one for prediction. These methods can be found in the [main.py](/code/main.py) file. For more information, please refer to the official [IGNNITION documentation.](https://ignnition.net/doc/)

#### Preparing the datasets
In order to properly feed the dataset to IGNNITION, the dataset must be in json format (you can get more information [here](https://ignnition.net/doc/generate_your_dataset/)). For this, we provide a [migrate.py](code/migrate.py) file that reads the provided datasets and transforms them to the proper format.

This file contains 5 parameters that can be defined:
* DATA_DIR: Directory where the file will read the dataset.
* DESTINATION_DIR: Directory where the processed JSON files will be stored.
* NUM_SAMPLES: Number of samples to process. If set to None, all the samples will be processed.
* JSON_MAX_SAMPLES: Number of max samples per JSON file.
* NUM_CORES: Number of cores to be used.


#### Training and evaluation
The [main.py](/code/main.py) file automatically trains and evaluates your model. This script trains during a max number of epochs, evaluating and saving the model for each one.

**IMPORTANT NOTE:** The execution can be stopped and resumed at any time. However, **if you want to start a new training phase you need to specify a new [output_path](code/train_options.yaml) directory (or empty the previous one)**.

Note that all the parameters needed for the execution (max number of epochs, steps per epoch...) can be changed in the [train_options](code/train_options.yaml#L30) file.

#### Prediction
The last thing the [main.py](/code/main.py) file does is to return and store the predictions in a python array that then can be used to store them, post-process them... 

Note that, if you only want to make predictions once the model is trained, you can simply remove the train and validate methods found in the [main.py](/code/main.py#L41) file.

## 'How to' guide to modify the code
### Transforming the input data
Now, the model reads the data as it is in the datasets. However, it is often highly recommended to apply some transformations (e.g. normalization, standardization, etc.) to the data in order to facilitate the model to converge during training.
In the [main.py](code/main.py) module you can find a function called [normalization(feature, feature_name)](code/main.py#L25), where the feature variable represents the predictors used by the model and the feature_name variable the name of the variable.
For example, if you want to apply a logarithmic transformation over the 'delay' variable, you can use the following code:
```
def normalization(feature, feature_name):
    if feature_name == 'delay':
        feature = np.log(feature)
    return feature
```

In this particular case, we transformed the target variable which in this case is the delay. This implies that all the metrics will be printed using the normalized data. If you want to obtain the specified metrics denormlized, you simply need to define the denormalization inside the [denormalization(feature, feature_name)](code/main.py#L30). Here an example:
```
def denormalization(feature, feature_name):
    if feature_name == 'delay':
        feature = np.exp(feature)
    return feature
```

### Adding new features to the hidden states
Currently, the model only considers the 'bandwith' variable to initialize the initial state of paths.
If we take a look into the [model_description.yaml](code/model_description.yaml) module, we can see how the [state initialization](code/model_description.yaml#L16) is done:
```
entities:
- name: link
  state_dimension: 32
  initial_state:
    - type: build_state
      input: [$capacity]

- name: path
  state_dimension: 32
  initial_state:
    - type: build_state
      input: [$traffic]
```
For example, if you also want to add the packets transmitted to the paths' initial states, you can easily do so by changing the code to:
```
entities:
- name: link
  state_dimension: 32
  initial_state:
    - type: build_state
      input: [$capacity]

- name: path
  state_dimension: 32
  initial_state:
    - type: build_state
      input: [$traffic, $packets]
```

### Working with Queue Occupancy
One important difference between the training and validation datasets of the challenge is in the delay values, which in RouteNet are the output labels of the model. Particularly, we can observe that delay values in larger networks (i.e., in the validation dataset) are quite smaller than in the smaller networks of the dataset. This is mainly because the validation dataset contains topologies with larger link capacities. As a result, packets experience less delay along their paths. As an example, the figure below shows the distribution of delays (in log scale) taking a representative set of samples of both datasets:

<p align="center"> 
  <img src="/assets/log_delay.png" width="600" alt>
</p>

This may represent a fundamental problem for the neural network model, as it would need to produce new values considerably out of the distribution with respect to the samples observed in the training dataset. 
One approach to circumvent this problem would be to predict an indirect metric that keeps a similar distribution in the training and validation datasets, and then infer the final path delays with a simple post-processing at the end. For example, queue occupancy always ranges in [0, 1], and represents the average utilization of the queue (in number of packets) for a given sample. To do so, first we can adapt RouteNet description as follows, to predict in this case the queue occupancy. To do so, you need to modify the [readout function](/code/model_description.yaml#L63) found in the [model_description.yaml](/code/model_description.yaml) and change this line:
```
readout:
- type: neural_network
  input: [path]
  nn_name: readout_model
  output_label: [$delay]
```
To this one:
```
readout:
- type: neural_network
  input: [link]
  nn_name: readout_model
  output_label: [$delay]
```

Finally, as the outputs required in the challenge are the per-path mean delays, it is possible to make a final post-processing to infer them from the avg. queue occupancy estimated by RouteNet. Particularly, path delays can be estimated as the linear combination of delays on each queue that form the path, considering the avg. queue occupation, the queue size, and the capacity of the outgoing link. The delay of a queue can be computed by dividing the avg. number of packets in the queue (predicted-queue-occupancy * queue size), by the capacity of the outgoing link of the queue.

### Available features
In the previous example we could directly include the packets transmitted ($packets) into the paths’ hidden states. This is because this implementation provides some dataset features that are already processed from the dataset. This can be found in the [network_to_hypergraph](/code/migrate.py#L38) function in the [migrate.py](/code/migrate.py) file, where the following features are included:
* 'bandwidth': This feature represents the bitrate (bits/time unit) of all the src-dst paths (This is obtained from the traffic_matrix[src,dst][′Flows′][flow_id][‘AvgBw’] values of all src-dst pairs using the [DataNet API]((https://github.com/knowledgedefinednetworking/datanetAPI/tree/challenge2020)))
* 'packets': This feature represents the rate of packets generated (packets/time unit) of all the src-dst paths (This is obtained from the traffic_matrix[src,dst][′Flows′][flow_id][‘PktsGen’] values of all src-dst pairs using the [DataNet API]((https://github.com/knowledgedefinednetworking/datanetAPI/tree/challenge2020)))
* 'capacity': This feature represents the link capacity (bits/time unit) of all the links found on the network (This is obtained from the topology_object[node][adj][0]['bandwidth'] values of all node-adj pairs using the [DataNet API]((https://github.com/knowledgedefinednetworking/datanetAPI/tree/challenge2020)))

Note that there are additional features in our datasets that are not included in this TensorFlow Data structure. However, they can be included processing the data with the dataset API and converting it into tensors. For this, you need to modify the [network_to_hypergraph](/code/migrate.py#L38) function in the [migrate.py](/code/migrate.py) file. Please, refer to the [API documentation]( https://github.com/BNN-UPC/datanetAPI/tree/challenge2021) of the datasets to see more details about all the data included in our datasets.

**IMPORTANT NOTE:** For the challenge, consider that variables under the structures performance_matrix and port_stats cannot be used as inputs of the model, since they will not be available in the final test set (see more info at the [API documentation]( https://github.com/BNN-UPC/datanetAPI/tree/challenge2021).

### Hyperparameter tunning
If you also want to modify or even add new hyperparameters, you can do so by modifying the [[HYPERPARAMETERS]](code/config.ini#L9) section in the [config.ini](code/config.ini) file.

## Credits
This project would not have been possible without the contribution of:
* [Miquel Ferriol Galmés](https://github.com/MiquelFerriol) - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* [Jose Suárez-Varela](https://github.com/jsuarezv) - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* [David Pujol](https://github.com/dpujol14) -  Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* [Albert López](https://github.com/albert-lopez) - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* [Paul Almasan](https://github.com/paulalmasan) - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* Adrián Manco Sánchez - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* Víctor Sendino Garcia - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* [Pere Barlet Ros](https://github.com/pbarlet) - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* Albert Cabellos Aparicio - Barcelona Neural Networking center, Universitat Politècnica de Catalunya

## Mailing List
If you have any doubts, or want to discuss anything related to this repository, you can send an email to the mailing list [challenge2021@bnn.upc.edu]( https://mail.bnn.upc.edu/cgi-bin/mailman/listinfo/challenge2021)). Please, note that you need to subscribe to the mailing list before sending an email [[link]( https://mail.bnn.upc.edu/cgi-bin/mailman/listinfo/challenge2021)].
## License
See [LICENSE](LICENSE) for full of the license text.
```
Copyright Copyright 2021 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
