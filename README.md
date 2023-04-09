[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KAUST-Academy/introduction-to-graph-neural-networks/HEAD)

# Introduction to Graph Neural Networks

There is strong demand, both globally and locally in KSA, for deep learning (DL) skills and expertise to solve challenging business problems. This course will help learners build capacity in the core DL tools and methods used in the computer vision field and enable them to develop their own graph neural network (GNN) applications. This course covers the basic theory behind key GNN algorithms but the majority of the focus is on building GNN applications using [PyTorch](https://pytorch.org/).

## Learning Objectives

The primary learning objective of this course is to provide students with practical, hands-on experience with state-of-the-art machine learning and deep learning tools that are widely used in the industries applying GNNs.

This course covers relevant portions of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) and [Machine Learning with PyTorch and Scikit-Learn](https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312). The following topics will be discussed.

* Basic ideas of Convolutional Neural Networks (CNNs)
* Basis ideas of Graph Neural Networks (GNNs)
* Applications of GNNs

## Lessons

The lessons are organizes into modules with the idea that they can taught somewhat independently to accommodate specific audiences. It is assumed that learners will have sufficient background in the basics of DL equivalent to having taken [Introduction to Deep Learning](https://github.com/KAUST-Academy/introduction-to-deep-learning).

### Module 0: Review of Deep Learning Fundamentals

Materials should be completed prior to arriving at any in-person training.

### Module 1: [Introduction to GNNs](https://kaust-my.sharepoint.com/:p:/g/personal/pughdr_kaust_edu_sa/EUFMLrIgoptFk3iG0IJOwsUBFMjEmqldB5WnEqZwUilKmg?e=JZF5ST)

* Review of Deep Learning fundamentals.  
* The morning session will focus on the theory behind Graph Neural Networks (GNNs) by covering relevant portions of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) and [Machine Learning with PyTorch and Scikit-Learn](https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312).    
* The afternoon session will focus on applying the techniques learned in the morning session using [PyTorch](https://pytorch.org/), followed by a short assessment on the Kaggle data science competition platform.

| **Tutorial** | **Open in Google Colab** | **Open in Kaggle** |
|--------------|:------------------------:|:------------------:|
| Introduction to GNNs with PyTorch | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/04a-intro-to-graph-neural-networks-with-pytorch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/04a-intro-to-graph-neural-networks-with-pytorch.ipynb) | 
| Introduction to GNNs with PyTorch Geometric | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/04b-intro-to-graph-neural-networks-with-pytorch-geometric.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/KAUST-Academy/introduction-to-deep-learning/blob/master/notebooks/04b-intro-to-graph-neural-networks-with-pytorch-geometric.ipynb) | 

## Assessment

Student performance on the course will be assessed through participation in a Kaggle classroom competition. 

# Repository Organization

Repository organization is based on ideas from [_Good Enough Practices for Scientific Computing_](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510).

1. Put each project in its own directory, which is named after the project.
2. Put external scripts or compiled programs in the `bin` directory.
3. Put raw data and metadata in a `data` directory.
4. Put text documents associated with the project in the `doc` directory.
5. Put all Docker related files in the `docker` directory.
6. Install the Conda environment into an `env` directory. 
7. Put all notebooks in the `notebooks` directory.
8. Put files generated during cleanup and analysis in a `results` directory.
9. Put project source code in the `src` directory.
10. Name all files to reflect their content or function.

## Building the Conda environment

After adding any necessary dependencies that should be downloaded via `conda` to the 
`environment.yml` file and any dependencies that should be downloaded via `pip` to the 
`requirements.txt` file you create the Conda environment in a sub-directory `./env`of your project 
directory by running the following commands.

```bash
export ENV_PREFIX=$PWD/env
mamba env create --prefix $ENV_PREFIX --file environment.yml --force
```

Once the new environment has been created you can activate the environment with the following 
command.

```bash
conda activate $ENV_PREFIX
```

Note that the `ENV_PREFIX` directory is *not* under version control as it can always be re-created as 
necessary.

For your convenience these commands have been combined in a shell script `./bin/create-conda-env.sh`. 
Running the shell script will create the Conda environment, activate the Conda environment, and build 
JupyterLab with any additional extensions. The script should be run from the project root directory 
as follows. 

```bash
./bin/create-conda-env.sh
```

### Ibex

The most efficient way to build Conda environments on Ibex is to launch the environment creation script 
as a job on the debug partition via Slurm. For your convenience a Slurm job script 
`./bin/create-conda-env.sbatch` is included. The script should be run from the project root directory 
as follows.

```bash
sbatch ./bin/create-conda-env.sbatch
```

### Listing the full contents of the Conda environment

The list of explicit dependencies for the project are listed in the `environment.yml` file. To see 
the full lost of packages installed into the environment run the following command.

```bash
conda list --prefix $ENV_PREFIX
```

### Updating the Conda environment

If you add (remove) dependencies to (from) the `environment.yml` file or the `requirements.txt` file 
after the environment has already been created, then you can re-create the environment with the 
following command.

```bash
$ mamba env create --prefix $ENV_PREFIX --file environment.yml --force
```

## Using Docker

In order to build Docker images for your project and run containers with GPU acceleration you will 
need to install 
[Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/), 
[Docker Compose](https://docs.docker.com/compose/install/) and the 
[NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker).

Detailed instructions for using Docker to build and image and launch containers can be found in 
the `docker/README.md`.
