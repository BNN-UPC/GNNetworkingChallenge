# Graph Neural Networking Challenge 2022: Improving Network Digital Twins through Data-centric AI

For more information about this challenge go to: https://bnn.upc.edu/challenge/gnnet2022/

## Repository Structure
- RouteNet_Fermi: module used to train and evaluate the GNN model for the challenge. The module's API is described below. **DO NOT MODIFY ITS CONTENTS**, instead use its API to train and evaluate the model. The GNN model used is known as RouteNet-Fermi, for more information about it please visit https://bnn.upc.edu/challenge/gnnet2022/#gnn-model.
- validation_dataset: folder which contains the validation dataset. **DO NOT MODIFY, DO NOT USE THESE SAMPLES FOR TRAINING**
- [quickstart.ipynb](quickstart.ipynb): jupyter notebook which introduces the steps necessary to participate in the challenge (dataset generation & model training). **START HERE!**
- [input_parameters_glossary.ipynb](input_parameters_glossary.ipynb): jupyter notebook which contains all the information about how to build the dataset, which parameters can be modified, and what constitutes a valid dataset.
- [dataset_visualization.ipynb](dataset_visualization.ipynb): jupyter notebook with some code you can use to better visualize and analyze the generated datasets.
- [training_dataset_constraints.md](training_dataset_constraints.md): markdown file which lists all the contraints that valid datasets must follow.
- README.md

Note: the simulator to generate samples is *NOT* included in this repository. Instead, it is uploaded at dockerhub, and will be downloaded automatically when trying to execute it. For details of how this is done, follow the [quickstart notebook](quickstart.ipynb). For more information about the simulator please visit https://bnn.upc.edu/challenge/gnnet2022/#network-simulator-and-dataset.

## Set up

### Setting up the docker enviroment

Either *Docker Engine* or *Docker Desktop* must be installed in your machine in order to execute the packet simulator and being able to generate the training samples. We refer the following links so you can install docker in your machine according to your OS:

- Docker Desktop: https://docs.docker.com/desktop/
- Docker Engine: https://docs.docker.com/engine/

### Setting up the Python enviroment

In order to build and train the model a working *Python 3.9* enviroment is required with, at least, the following packages:
- [networkx](https://networkx.org/) 2.8.11
- [tensorflow](https://www.tensorflow.org/) 2.7
- [PyYAML](https://pyyaml.org/) 6.0
- [Jupyter Notebook](https://jupyter.org/) 6.4.11 (For running the provided notebooks)

For the "dataset_visualization.ipynb" notebook two additional packages are needed for generating the graphs:
- [matplotlib](https://matplotlib.org/) 3.5.2
- [astropy](https://www.astropy.org/) 5.1

We recommend setting up a clean python virtual enviroment for this project (e.g. virtualenv or conda). You can install all the required packages with the following pip command:

```
pip install notebook==6.4.11 PyYAML==6.0 tensorflow==2.7 networkx==2.8.1 matplotlib==3.5.2 astropy==5.1
```

Note that for training the model the usage of GPU acceleration is **NOT** required. The training process has been designed so it can be run in personal computers and laptops in a reasonable amount of time even when only using the CPU. Nevertheless, for more information on how to install Tensorflow for GPU, check out https://www.tensorflow.org/install/gpu.

## RouteNet_Fermi's API

- ```main(train_path, final_evaluation = False, ckpt_dir="./modelCheckpoints")```

    Trains and evaluates the model with the provided dataset.
    The model will be trained for 20 epochs.
    At each epoch a checkpoint of the model will be generated and stored at the folder ckpt_dir which will be created automatically if it doesn't exist already.
    Training the model will also generate logs at "./logs" that can be opened with tensorboard.

    Parameters:
    - ```train_path```: Path to the training dataset
    - ```final_evaluation```: If True after training the model will be validated using all of the validation dataset, by default False
    - ```ckpt_dir```: Relative path (from the repository's root folder) where the model's weight will be stored, by default "./modelCheckpoints"

- ```evaluate(ckpt_path)```

    Loads model from checkpoint and trains the model.

    Parameters:
    - ```ckpt_path```: Path to the checkpoint. Format the name as it was introduced in tf.keras.Model.load_weights.


-----
For more information about the this challenge go to: https://bnn.upc.edu/challenge/gnnet2022/

For more information about the ITU AI/ML in 5G Challenge go to: https://aiforgood.itu.int/about-ai-for-good/aiml-in-5g-challenge/
