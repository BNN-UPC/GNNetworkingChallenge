# Graph Neural Networking Challenge 2023: Creating a Network Digital Twin with Real Network Data

For more information about this challenge go to: https://bnn.upc.edu/challenge/gnnet2023

For more information about the ITU AI/ML in 5G Challenge go to: https://aiforgood.itu.int/about-ai-for-good/aiml-in-5g-challenge/

Remember that the full datasets must be downloaded separately at: https://bnn.upc.edu/challenge/gnnet2023/dataset


- [Graph Neural Networking Challenge 2023: Creating a Network Digital Twin with Real Network Data](#graph-neural-networking-challenge-2023-creating-a-network-digital-twin-with-real-network-data)
  - [Repository structure](#repository-structure)
  - [Quickstart](#quickstart)
    - [Python environment](#python-environment)
    - [Downloading the dataset and preprocessing the dataset](#downloading-the-dataset-and-preprocessing-the-dataset)
    - [(Alternative) work with the provided pre-processed datasets](#alternative-work-with-the-provided-pre-processed-datasets)
    - [Training and evaluating the model](#training-and-evaluating-the-model)
  - ["How-to" guide for modifying the code](#how-to-guide-for-modifying-the-code)
    - [Default features](#default-features)
    - [Extracting new features](#extracting-new-features)
      - [**Example: how to extract the Inter-Packet Gap**](#example-how-to-extract-the-inter-packet-gap)
    - [More processing after data\_generator.py](#more-processing-after-data_generatorpy)
    - [Changing the model's hyperparameters](#changing-the-models-hyperparameters)
      - [**Changing the the size of the internal embeddings**](#changing-the-the-size-of-the-internal-embeddings)
      - [**Modify the internal MLPs and RNNs of the model**](#modify-the-internal-mlps-and-rnns-of-the-model)
      - [**Changing how and which features are normalized**](#changing-how-and-which-features-are-normalized)
      - [**Changing the training process hyperparameters**](#changing-the-training-process-hyperparameters)
  - [Credits](#credits)
  - [Mailing list](#mailing-list)


## Repository structure

- data: folder containing the code needed to proceprocess the samples. It also includes versions of the datasets with the baseline's procecessing already applied.
- [models.py](models.py): python module which contain the two baseline models, one for each dataset
- [train.py](train.py): python script that can be used to train the baseline projects

## Quickstart

### Python environment

In order to work with the dataset a working Python 3.9+ environment is required with, at least, the following packages:

- [networkx](https://networkx.org/) 3.0
- [numpy](https://numpy.org/) 1.24.2


To work with the baseline models, [tensorflow](https://www.tensorflow.org/) 2.11.1 is also required.

We recommend setting up a clean python virtual environment for this project (e.g. virtualenv or conda). You can install all the required packages with the following pip commands (the latter only to install tensorflow):

```
pip install networkx==3.0 numpy==1.24.2
```
```
pip install tensorflow==2.11.1
```


If you have issues while running the code, please verify that the *exact* packages versions are being used. Python versions 3.8.10 and 3.9.16 have proven to work.

### Downloading the dataset and preprocessing the dataset

Visit the following link to download the dataset and see the instructions on how to explore it: https://bnn.upc.edu/challenge/gnnet2023/dataset

*WARNING*: these datasets are large (458 GB and 290 GB for the CBR+MB and MB datasets, respectively), make sure you have enough disc space beforehand.

Once you download the datasets, you may apply the default pre-processing by running the [data_generator.py](data/data_generator.py) script:

```
python data/data_generator.py --input-dir "/path/to/cbr+mb/dataset" --output-dir data/data_cbr_mb

python data/data_generator.py --input-dir "/path/to/mb/dataset" --output-dir data/data_mb
```

### (Alternative) work with the provided pre-processed datasets

Alternatively, you can work with the preprocessed datasets we have already provided within this repository.
- CBR+MB dataset:
  - In its entirety: data/data_cbr_mb
  - Split into 5 folds, each divided into training / validation, for 5-fold cross validation: data/data_cbr_mb_cv
- MB dataset:
  - In its entirety: data/data_mb
  - Split into 5 folds, each divided into training / validation, for 5-fold cross validation: data/data_cbr_cv

Note that these datasets are much smaller (~ 100 MBs total), but you will not be able to extract new features from the raw data. It is meant to replicate the baseline results and as an alternative to participants that may not have the resources to download the datasets.

### Training and evaluating the model

We have included all the code to train and evaluate the baseline models in the following python files:
- [models.py](models.py): Contains the two baseline model architectures (one meant for each dataset)
- [train.py](train.py): Script to train and evaluate models

By using the [train.py](train.py) script, we can train the models with the different datasets. By default, the following calls are supported:
```
# Train the baseline model using the CBR+MB dataset
python train.py -ds CBR+MB

# Train the baseline model using the CBR+MB dataset through 5-fold cross validation
python train.py -ds CBR+MB -cfv

# Train the baseline model using the MB dataset
python train.py -ds MB

# Train the baseline model using the MB dataset through 5-fold cross validation
python train.py -ds MB -cfv
```

When running by default, the checkpoints with the weights will be stored at the *ckpt/* directory, and the tensorboard logs at the *tensorboard/* directory.

In order to have more control when training the models, you can import the `train_and_evaluate` function and adjust its argument.
```
from train import train_and_evaluate
```

## "How-to" guide for modifying the code

This section is meant in showing you how can you modify the code provided in the repository to generate your own solutions.

### Default features

The features used by default in the baseline are the following:
-  `flow_traffic`: the average traffic bandwidth per flow in bps
-  `flow_packets`: the number of generated packets per flow
-  `flow_packet_size`: the size of the generated packets per flow
-  `flow_type`: two-dimensional one-hot encoded feature used to identify the flow type of each flow
   -  `[1, 0]` indicates the flow is a Constant Bit Rate (CBR) flow
   -  `[0, 1]` indicates the flow is a Multi Burst (MB) flow
- `flow_length`: length of the physical path followed by each flow
- `link_capacity`: for each link, it indicates its bandwidth in bps
- `link_to_path`: for each flow, it indicates the links forming its path, in order
- `path_to_link`: for each link, it lists the flows that traverse it. It also includes the position of the link in each flow's path. For a given link the same flow can appear more than once if the link is traversed more than one in the same flow path

The target metric is `flow_delay`, the mean packet delay per flow, in mbps.

From these features, `flow_traffic`, `flow_packets`, `flow_packet_size` and `link_capacity` are normalized using min-max normalization.

Additionally, the following feature is computed at run time during training:
- `load`: for each link, the expected load is computed by combining the `flow_traffic`and `path_to_link` features.

Finally, three additional features are extracted from data in order to identify the samples. These are not used in the model itself, but are left for more insight during debugging and to correctly identify each node:
- `sample_file_name`
- `sample_file_id`
- `flow_id`

### Extracting new features

The [data_generator.py](data/data_generator.py) script works by loading the dataset from disc, and processing each sample individually into a usable format by tensorflow. To do so it uses three functions:
- `_get_network_decomposition`: this function takes a sample in the format provided by the dataset's API (named DatanetAPI), and returns a sample in a format accepted by tensorflow models
- `_generator`: this function defines a generator which loads the datasets and processes its samples one by one using `_get_network_decomposition`. It also filters out unsuitable samples.
- `input_fn`: wrapper function that uses `_generator` and `tensorflow.data.Dataset.from_generator` to transform the raw dataset into a `tensorflow.data.Dataset`.

In order to add additional features to the model, changes must be done at three distinct points of the [data_generator.py](data/data_generator.py) script:
- Extract the new feature from the sample (lines 91 - 141).
- Ensure that the new feature is returned by the `_get_network_decomposition` function (lines 198 - 227).
- Change the signature of the tf.data.Dataset to include the new feature inside the `input_fn` function (lines 276-293).

You may extract any feature that may be provided by the DatanetAPI, with the **exception being performance matrix and the individual packet delays found at the packet info data structure**, as these **will not** be provided with the test dataset.

For more information about the DatanetAPI, please check the README file inside the dataset's compressed directory, or the online documentation at https://github.com/BNN-UPC/datanetAPI/tree/challenge2023.

#### **Example: how to extract the Inter-Packet Gap**

For better understanding on how a new feature can be added, we will proceed to explain how the Inter-Packet Gap (IPG) can extracted and added to the samples as features. The IPG is measured as the time past between two consecutive packets.

First, we need to modify the implementation of `_get_network_decomposition` so that the IPG is extracted. Currently, the function already extracts the network topology, traffic matrix, routing matrix and performance matrix from the sample. However, the IPG is extracted from the packet-level information, so we need to load the packet info matrix:
```python
# sample is a Sample instance, an input of the _get_network_decomposition function.
network_topology = sample.get_physical_topology_object()
traffic_matrix = sample.get_traffic_matrix()
physical_path_matrix = sample.get_physical_path_matrix()
performance_matrix = sample.get_performance_matrix()
# V NEW CHANGE HERE V
packet_info_matrix = sample.get_pkts_info_object()
```

The matrix contains the packet traces for all the flows inside. In order to iterate through all the flows, we then use the traffic matrix to identify which routers have flows between them, and how many of them (*NOTE*: this code is already present in lines 91-96 at [data_generator.py](data/data_generator.py)):
```python
for src, dst in filter(
    lambda x: traffic_matrix[x]["AggInfo"]["AvgBw"] != 0
    and traffic_matrix[x]["AggInfo"]["PktsGen"] != 0,
    permutations(range(len(traffic_matrix)), 2),
):
    for f_id in range(len(traffic_matrix[src, dst]["Flows"])):
        flow_packet_info = packet_info_matrix[src, dst][0][f_id]
```
Here, the outer loop iterates through all router pairs, and the condition represented by the lambda function ensures that there is at least one flow between them. The inner loop iterates across all flows that start at `src` and end at `dst`. In the final line we access the packet-level information for the given flow.

*Note: for more details about how to access the packet_info_matrix and how to interpret the packet-level information present, please check the DatanetAPI's README file inside the dataset's directory or the online documentation at https://github.com/BNN-UPC/datanetAPI/tree/challenge2023*

Now, the `flow_packet_info` is defined as a list where each element is the information related to each individual captured packet. The elements of the list are ordered using the timestamp of the packet's creation. Each element is a tuple containing this precise timestamp and the delay suffered by the packet during its transmission. For the purpose of obtaining the IPG, we only need the former:
```python
packet_timestamps = np.array([float(x[0]) for x in flow_packet_info])
```

With it, we can obtain the sequence of IPGs for that flow:
```python
ipg = packet_timestamps[1:] - packet_timestamps[:-1]
```

While we can use the sequence of IPGs as a feature, we may present the information in a more reduced form. For example, we can extract the mean, the variance, and even all the percentiles. In whatever changes we perform, we can add it with the rest of flow features (lines 106-123 at [data_generator.py](data/data_generator.py)):

```python
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
    # V NEW CHANGE HERE V
    "ipg_mean": np.mean(ipg),
    "ipg_var": np.var(ipg),
    "ipg_percentiles": np.percentile(ipg, range(101)),
}
```

The next step is making sure that the IPG is included in the features returned by the `_get_network_decomposition` function (at lines 180 - 209 at [data_generator.py](data/data_generator.py)):
```python
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
        # V NEW CHANGE HERE V
        "flow_ipg_mean": np.expand_dims(
            [flow["ipg_mean"] for flow in ordered_flows], axis=1
        ),
        "flow_ipg_var": np.expand_dims(
            [flow["ipg_var"] for flow in ordered_flows], axis=1
        ),
        "flow_ipg_percentiles": [flow["ipg_percentiles"] for flow in ordered_flows],
        # ^ NEW CHANGE HERE ^
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
```

*Note*: barring some exceptions like the flow delay and flow length, new features are represented with a two dimensional matrix. The idea behind it is that later on these features will be concatenated into a matrix in which each row represents the features associated to each flow/link. This means that for features that are single values, such as the `ipg_mean`, must have their representation expanded from an n-sized vector to a (n, 1) sized matrix to ease their concatenation. Other features that already are represented through  a matrix, such as `ipg_percentiles`, do not need to be adjusted.

The final step is to modify the dataset signature inside the `input_fn` function. The signature is a dict which describes the format of the samples within it. Hence, any new feature must be correctly represented within the signature. We can add the IPG-based features as follows (lines 258-275 at [data_generator.py](data/data_generator.py)):

```python
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
        # V NEW CHANGE HERE V
        "flow_ipg_mean": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "flow_ipg_var": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "flow_ipg_percentiles": tf.TensorSpec(shape=(None, 101), dtype=tf.float32),
        # ^ NEW CHANGE HERE ^
    },
    tf.TensorSpec(shape=None, dtype=tf.float32),
)
```

After this final step, we can run again the [data_generator.py](data/data_generator.py) script to process the dataset again and work with the new feature. Of course, once the feature is extracted remember to change the model for it to be used.

### More processing after data_generator.py

The easiest way to implement changes to the dataset is by using tensorflow.data.Dataset's `map` function. This allows you to define a function that takes as input each sample and its label and modify it. For example, this sample code shows how you can change the samples so that label is replaced by its logarithm:
```python
import tensorflow as tf

def transformation(x, y):
    return x, tf.math.log(y)

ds_transformed = ds.map(transformation)
```

The advantage of this method is that it quicker than modifying the [data_generator.py](data/data_generator.py) script and re-generating the dataset. However, this come at the cost that the changes must be applied every time the model is trained, and that you are limited by the features already given by the original dataset.

### Changing the model's hyperparameters

You can modify any of the baseline models by modifying their code, which can be found inside the [models.py](models.py) python file. **NOTE**: unless you modify the training and evaluation pipeline, the models must follow the following conditions:
- Have a string attribute called `name` to identify the model's architecture
- Have a set attribute named `min_max_scores_fields` and a function called `set_min_max_scores` to add min-max normalization to the samples in the model
  - `min_max_scores_fields` is used to identify which of the input fields must be normalized
  - `set_min_max_scores` will be used so that the model store the min-max scores. For more details, refer to its implementation

Possible changes include, but aren't limited to:

#### **Changing the the size of the internal embeddings**

Changing the size of the internal embeddings is as simple as modifying the `path_state_dim` and `link_state_dim` attributes of the models:
```python
def __init__(self, override_min_max_scores=None, name=None):
    super(Baseline_cbr_mb, self).__init__()

    self.iterations = 8
    # self.path_state_dim = 64
    self.path_state_dim = 128
    self.link_state_dim = 64
```

#### **Modify the internal MLPs and RNNs of the model**

The Baseline model counts with the following MLPs:
- `flow_embedding`: used to generate an initial representation of flows taking as input the flow's attributes
- `link_embedding`: used to generate an initial representation of links taking as input the flow's attributes
- `readout_path`: takes as input the path sequence state and generates delay predictions

The Baseline model counts with the following RNNs:
- `path_update`: used to update the flow states during message passing
- `link_update`: used to update the link states during the message passing

All of these functions are defined inside the `__init__` function and can be modified at will. For example, by default weight regularization is not included, but it can be added as an attribute when defining the MLPs:
```python
self.flow_embedding = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=5),
        tf.keras.layers.Dense(
            self.path_state_dim,
            activation=tf.keras.activations.relu,
            kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
            bias_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
        ),
        tf.keras.layers.Dense(
            self.path_state_dim,
            activation=tf.keras.activations.relu,
            kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
            bias_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
        ),
    ],
    name="PathEmbedding",
)
self.link_embedding = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=2),
        tf.keras.layers.Dense(
            self.link_state_dim,
            activation=tf.keras.activations.relu,
            kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
            bias_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
        ),
        tf.keras.layers.Dense(
            self.link_state_dim,
            activation=tf.keras.activations.relu,
            kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
            bias_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
        ),
    ],
    name="LinkEmbedding",
)
self.readout_path = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(None, self.path_state_dim)),
        tf.keras.layers.Dense(
            self.link_state_dim // 2,
            activation=tf.keras.activations.relu,
            kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
            bias_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
        ),
        tf.keras.layers.Dense(
            self.link_state_dim // 4,
            activation=tf.keras.activations.relu,
            kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
            bias_regularizer= tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
        ),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
    ],
    name="PathReadout",
)
```

#### **Changing how and which features are normalized**

Currently, min-max normalization is implemented in the model as follows: when a model is instantiated, the min-max normalization values are computed and passed to the model in the following line:
```
model.set_min_max_scores(get_min_max_dict(ds_train, model.min_max_scores_fields))
```
- `model.min_max_scores_fields` is a set that indicates the name of the fields that need min-max normalization
- `get_min_max_dict` is a function in [train.py](train.py) that obtains the min-max weights
- `model.set_min_max_scores` is a function that verifies that the format of the min-max weights is correct, and stores them inside the attribute `model.min_max_scores` so they can be used within `model.call`.

Setting the min-max weights is a requisite for the models to function properly.

If you wish to compute the min-max normalization of a new variable, you simply must include the variable name in `model.min_max_scores_fields`. Then, inside the `model.call` function, you must ensure that you normalize the features before introducing them:
```python
class Baseline_cbr_mb(tf.keras.Model):
    min_max_scores_fields = {
        "flow_traffic",
        "flow_packets",
        "flow_packet_size",
        "link_capacity",
    }
    ...

    # Inside call()
    flow_traffic_normalized = (flow_traffic - self.min_max_scores["flow_traffic"][0]) * self.min_max_scores["flow_traffic"][1]
```

If you wish to use another type of normalization (e.g. z-scores), remember to define a new function similar to `get_min_max_dict`, but with the desired normalization technique, as well as modifying how variables are normalized within the model itself.

#### **Changing the training process hyperparameters**

One can make changes to the training process hyperparameters by running a script that imports the `train_and_evaluate` function from [train.py](train.py) in order to have more control over them.

For example, in order to train a model using:
- SGD instead of Adam as an optimizer
- 150 epochs instead of 100
- Mean Square Error Loss rather than Mean Absolute Percentage Error
```python
from train import get_default_callbacks, train_and_evaluate
...

trained_model, evaluation = train_and_evaluate(
    ds_path,
    model,
    tf.keras.optimizers.SGD(learning_rate=0.001),
    [tk.keras.losses.MeanSquareError],
    [tf.keras.losses.MeanAbsolutePercentageError()],
    get_default_callbacks(),
    epochs=150
)
```

*NOTE:* while you are free to choose the loss function you believe it is best, your solution will be evaluated using the MAPE.

If there is another aspect of the pipeline that you wish to change, but it is not covered by the method's attribute, you are encouraged to make any changes to the [train.py](train.py) you wish.
## Credits
This project would not have been possible without the contribution of:
* [Carlos Güemes Palau](https://github.com/CarlosGuemS) - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* [Miquel Ferriol-Galmés](https://github.com/MiquelFerriol) - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* [Albert López](https://github.com/albert-lopez) - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* [Pere Barlet Ros](https://github.com/pbarlet) - Barcelona Neural Networking center, Universitat Politècnica de Catalunya
* Albert Cabellos Aparicio - Barcelona Neural Networking center, Universitat Politècnica de Catalunya

## Mailing list

If you have any doubts, or want to discuss anything related to this repository, you can send an email to the mailing list [challenge2023@bnn.upc.edu](TODO). Please, note that you need to subscribe to the mailing list before sending an email [[link](TODO)].