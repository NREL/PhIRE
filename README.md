# Towards Representation Learning for Atmospheric Dynamics (AtmoDist)

> The prediction of future climate scenarios under anthropogenic forcing is critical to understand climate change and to assess the impact of potentially counter-acting technologies. Machine learning and hybrid techniques for this prediction rely on informative metrics that are sensitive to pertinent but often subtle influences. For atmospheric dynamics, a critical part of the climate system, no well established metric exists and visual inspection is currently still often used in practice. However, this "eyeball metric" cannot be used for machine learning where an algorithmic description is required. Motivated by the success of intermediate neural network activations as basis for learned metrics, e.g. in computer vision, we present a novel, self-supervised representation learning approach specifically designed for atmospheric dynamics. Our approach, called AtmoDist, trains a neural network on a simple, auxiliary task: predicting the temporal distance between elements of a randomly shuffled sequence of atmospheric fields (e.g. the components of the wind field from reanalysis or simulation). The task forces the network to learn important intrinsic aspects of the data as activations in its layers and from these hence a discriminative metric can be obtained. We demonstrate this by using AtmoDist to define a metric for GAN-based super resolution of vorticity and divergence. Our upscaled data matches both visually and in terms of its statistics a high resolution reference closely and it significantly outperform the state-of-the-art based on mean squared error. Since AtmoDist is unsupervised, only requires a temporal sequence of fields, and uses a simple auxiliary task, it has the potential to be of utility in a wide range of applications.

Original implementation of 

> *Hoffmann, Sebastian, and Christian Lessig. "Towards Representation Learning for Atmospheric Dynamics." arXiv preprint arXiv:2109.09076 (2021).* https://arxiv.org/abs/2109.09076

presented as part of the NeurIPS 2021 Workshop on [Tackling Climate Change with Machine Learning](https://www.climatechange.ai/events/neurips2021)

We would like to thank Stengel et al. for openly making available their implementation (https://github.com/NREL/PhIRE) of [Adversarial super-resolution of climatological wind and solar data](https://www.pnas.org/content/117/29/16805) on which we directly based the super-resolution part of this work.
___
### Requirements
* tensorflow 1.15.5
* pyshtools (for SR evaluation)
* pyspharm (for SR evaluation)
* h5py
* hdf5plugin
* dask.array

### Installation
`pip install -e ./`

This also makes available multiple command line tools that provide easy access to preprocessing, training, and evaluation routines. It's recommended to install the project in a virtual environment as to not polutte the global PATH.

___
### CLI Tools

The provided CLI tools don't accept parameters but rather act as a shortcut to execute the corresponding script files. All parameters controlling the behaviour of the training etc. should thus be adjusted in the script files directly. We list both the command-line command, as well as the script file the command executes.

* `rplearn-data` ([python/phire/data_tool.py](https://github.com/sehoffmann/AtmoDist/blob/master/python/phire/data_tool.py))
    * Samples patches and generates `.tfrecords` files from HDF5 data for the self-supervised representation-learning task.
* `rplearn-train` ([python/phire/rplearn/train.py](https://github.com/sehoffmann/AtmoDist/blob/master/python/phire/rplearn/train.py))
    * Trains the representation network. By toggling comments, the same script is also used for evaluation of the trained network.
* `phire-data` ([python/phire/data_tool.py](https://github.com/sehoffmann/AtmoDist/blob/master/python/phire/data_tool.py))
    * Samples patches and generates `.tfrecords` files from HDF5 data for the super-resolution task.
* `phire-train` ([python/phire/main.py](https://github.com/sehoffmann/AtmoDist/blob/master/python/phire/main.py))
    * Trains the SRGAN model using either MSE or a content-loss based on AtmoDist.
* `phire-eval` ([python/phire/evaluation/cli.py](https://github.com/sehoffmann/AtmoDist/blob/master/python/phire/evaluation/cli.py))
    * Evaluates trained SRGAN models using various metrics (e.g. energy spectrum, semivariogram, etc.). Generation of images is also part of this.

___
### Project Structure
* [python/phire](https://github.com/sehoffmann/AtmoDist/blob/master/python/phire)
    * Mostly preserved from the Stengel et al. implementation, this directory contains the code for the SR training. `sr_network.py` contains the actual GAN model, whereas `PhIREGANs.py` contains the main training loop, data pipeline, as well as interference procedure.
* [python/phire/rplearn](https://github.com/sehoffmann/AtmoDist/tree/master/python/phire/rplearn)
    * Contains everything related to representation learning task, i.e. AtmoDist. The actual ResNet models are defined in `resnet.py`, while the training procedure can be found in `train.py`.
* [python/phire/evaluation](https://github.com/sehoffmann/AtmoDist/tree/master/python/phire/evaluation)
    * Dedicated to the evaluation of the super-resolved fields. The main configuration of the evaluation is done in `cli.py`, while the other files mostly correspond to specific evaluation metrics.
* [python/phire/data](https://github.com/sehoffmann/AtmoDist/tree/master/python/phire/data)
    * Static data shipped with the python package.
* [python/phire/jetstream](https://github.com/sehoffmann/AtmoDist/tree/master/python/phire/jetstream)
    * WiP: Prediction of jetstream latitude as downstream task.
* [scripts/](https://github.com/sehoffmann/AtmoDist/tree/master/scripts/)
    * Various utility scripts, e.g. used to generate some of the figures seen in the paper.

___
### Preparing the data
AtmoDist is trained on vorticity and divergence fields from ERA5 reanalysis data. The data was directly obtained as spherical harmonic coefficients from model level 120, before being converted to regular lat-lon grids (1280 x 2560) using [`pyshtools`](https://shtools.github.io/SHTOOLS/index.html) (right now not included in this repository).

We assume this gridded data to be stored in a hdf5 file for training and evaluation respectively containing a single dataset `/data` with dimensions `C x T x H x W`. These dimensions correspond to time, channel (/variable), height, and width respectively. Patches are then sampled from this hdf5 data and stored in `.tfrecords` files for training.

In practice, these "master" files actually contained virtual datasets, while the actual data was stored as one hdf5 file per year. This is however not a hard requirement. The script to create these virtual datasets is currently not included in the repository but might be at a later point of time.

To sample patches for training or evaluation run `rplearn-data` and `phire-data`.

#### Normalization
Normalization is done by the [phire/data_tool.py](https://github.com/sehoffmann/AtmoDist/blob/master/python/phire/data_tool.py) script. This procedure is opaque to the models and data is only de-normalized during evaluation. The mean and standard deviations used for normalization can be specified using `DataSampler.mean, DataSampler.std, DataSampler.mean_log1p, DataSampler.std_log1p`. If specified as `None`, then these statistics will be calculated from the dataset using [`dask`](https://docs.dask.org/en/stable/array.html) (this will take some time).

___
### Training the AtmoDist model
1. Specify dataset location, model name, output location, and number of classes (i.e. max delta T) in [phire/rplearn/train.py](https://github.com/sehoffmann/AtmoDist/blob/master/python/phire/rplearn/train.py)
2. Run training using `rplearn-train`
3. Switch to evaluation by calling `evaluate_all()` and compute metrics on eval set
4. Find optimal epoch and calculate normalization factors (for specific layer) using `calculate_loss()`

___
### Training the SRGAN model
1. Specify dataset location, model name, AtmoDist model to use, and training regimen in [phire/main.py](https://github.com/sehoffmann/AtmoDist/blob/master/python/phire/main.py)
2. Run training using `phire-train`

___
### Evaluating the SRGAN models
1. Specify dataset location, models to evaluate, output location, and metrics to calculate in [phire/evaluation/cli.py](https://github.com/sehoffmann/AtmoDist/blob/master/python/phire/evaluation/cli.py)
2. Evaluate using `phire-eval`
3. Toggle the if-statement to generate comparing plots and data between different models and rerun `phire-eval`
