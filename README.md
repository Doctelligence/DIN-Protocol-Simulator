## Simulated Intelligence Protocol for Computing Clusters

This guide provides a framework for setting up a Federated Learning (FL) environment utilizing data records from any domain, ensuring privacy and decentralization.

### Initial Setup

- **Classification Tasks**: * For example, to start with a simple classification task, download an appropriate dataset, such as the chest x-ray pneumonia dataset from Kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

### Step 1: Environment Preparation
a) Activate a Virtual Python Learning Environment.
b) Create Simulated Data Records for Agents: In the script 'data-partition.py', set the 'no_agents' variable to the desired number of agents for your experiment.
   - Specify the directory for saving agent directories.
   - The script aims to evenly distribute negative and positive classes among agents.
c) Execute with `python data-partition.py`.

### Step 2: Model Training Setup
Navigate to the 'train.py' script.
a) Import a TensorFlow Lite Model (e.g., MobileNet version).
   - Find various model types at https://tfhub.dev/
   - Define 'module_selection' with model attributes (e.g., name, input size) and specify MODULE_HANDLE.
b) Set FL Variables in the 'train.py' script:
   1. Batch_size =
   2. num_epochs =
   3. max_rounds =
   4. no_agents =

### Step 3: Training Execution
a) Create a 'Global_model' subdirectory in the chosen location for saving the trained global models.
   - Models will be saved in sub-directories named after the FL variables for easy identification.
b) Run 'python train.py' (recommended: use an advanced computing cluster).

## Distributed Version Execution
### Step 1: Configuration Setup
Execute `export FL_CONFIG_FILE=/path/to/config/file` with the config file specifying:
   - `raw_data_root_dir`: Location of the dataset before partitioning.
   - `experiment_id`: Unique identifier for your experiment (e.g., "10_agents_5_rounds").
   - `directories`: Names and paths for data directories, including `root_dir`, `data_root_dir`, `test_dir`, `val_dir`, and `model_save_dir`.

### Step 2: Data Partitioning
Run the data partitioner with `python data-partition.py`, referencing `FL_CONFIG_FILE`.

### Step 3: MPI Requirement
Ensure `mpi4py` is installed, following [this guide](https://www.arc.ox.ac.uk/using-python-mpi-arc).

### Step 4: MPI Execution
Use MPI for distributed training: `mpirun -n <no_agents> python DistributedTrainer.py`.

## IPFS Private Node Setup
### Step 1: Node Configuration
Set up a private IPFS node as outlined [here](https://labs.eleks.com/2019/03/ipfs-network-data-replication.html) and run `ipfs daemon`.

### Steps 2 & 3: Configuration and Execution
Export the config file as in the Distributed Version section and execute `python ipfstrainer.py`.

## Private Blockchain and Computing Cluster Integration
### Step 1: Dependency Installation
Install necessary dependencies, including Geth. IT support may be required for installations.

### Step 2: Geth Initialization
Configure Geth on your computing cluster, utilizing separate terminals for setup and connection.

Run `geth/go:
srun -p interactive --pty /bin/bash
module spider geth --> geth: geth/2022
module load geth/2022` on your computing cluster
* OR `module load geth` 

### Step 3: MPI Environment Setup
Load necessary modules and activate the virtual environment for MPI.

Open two different windows for these commands.
Window 1
* `python regionNodeSetup.py 1 1`
* `python regionNodeSetup.py 1 0`
Window 2
* `python regionNodeConnector.py 1`

module purge
module load Anaconda3/2020.11
module load foss/2020a
source activate $YOURLOCATION/mpienv

### Step 4: Smart Contract Deployment
Proceed with deploying and executing smart contracts as part of your experimental setup.
* run `python submitContract.py`
* run `python smartContract_1.sol`
