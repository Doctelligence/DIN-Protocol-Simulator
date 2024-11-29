## Intelligence Protocol Simulator

- **Purpose**: Enable users to test an intelligence protocol for Federated Learning (FL) on distributed personal health records.
- **Features**: Offers a simulated environment for training machine learning algorithms on decentralized health data sources.
- **Testing Environment**: Designed for deployment in advanced research computing facilities.
- **Guiding Principles**: Ensures privacy, decentralization, and scalable health data integration throughout.
- **Intended Outcome**: Provides users with a platform to validate and refine the protocol’s efficiency in real-world health data scenarios.

### Initial Setup

- **Classification Tasks**: 📊 Start with a classification task by downloading a suitable dataset, such as the chest x-ray pneumonia dataset from Kaggle: [chest x-ray pneumonia dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

### Step 1: Environment Preparation

1. **Activate Environment**: 🖥️ Activate a Virtual Python Learning Environment.
2. **Create Simulated Data Records for Agents**:
   - Set the `no_agents` variable in `data-partition.py` to the number of agents for your experiment.
   - Specify the directory for saving agent directories.
   - This script evenly distributes negative and positive classes among agents.
   - Execute with `python data-partition.py`.

### Step 2: Model Training Setup

1. **Import Model**: 🏷️ Import a TensorFlow Lite Model (e.g., MobileNet):
   - Find various model types at [TensorFlow Hub](https://tfhub.dev/).
   - Define `module_selection` with model attributes and specify `MODULE_HANDLE`.
2. **Set FL Variables**:
   - Configure `Batch_size`, `num_epochs`, `max_rounds`, and `no_agents` in `train.py`.

### Step 3: Training Execution

1. **Create Model Directory**: 📁 Create a `Global_model` subdirectory for saving trained global models.
2. **Run Training**: 🚀 Execute `python train.py` (recommended: use an advanced computing cluster).

## Distributed Version Execution

### Step 1: Configuration Setup

1. **Export Config File**: 📜 Execute `export FL_CONFIG_FILE=/path/to/config/file` with the config file specifying dataset locations, experiment ID, and directory paths.

### Step 2: Data Partitioning

1. **Run Data Partitioner**: 🔄 Execute `python data-partition.py`, referencing `FL_CONFIG_FILE`.

### Step 3: MPI Requirement

1. **Install mpi4py**: 📦 Ensure `mpi4py` is installed, following [this guide](https://www.arc.ox.ac.uk/using-python-mpi-arc).

### Step 4: MPI Execution

1. **Distributed Training**: 🌐 Use MPI for distributed training: `mpirun -n <no_agents> python DistributedTrainer.py`.

## IPFS Private Node Setup

### Step 1: Node Configuration

1. **Set Up Node**: 🌐 Set up a private IPFS node and run `ipfs daemon` as outlined [here](https://labs.eleks.com/2019/03/ipfs-network-data-replication.html).

### Steps 2 & 3: Configuration and Execution

1. **Export Config**: 📁 Export the config file as described in the Distributed Version section.
2. **Run IPFS Trainer**: 🚀 Execute `python ipfstrainer.py`.

## Private Blockchain and Computing Cluster Integration

### Step 1: Dependency Installation

1. **Install Dependencies**: 🔧 Install necessary dependencies, including Geth.

### Step 2: Geth Initialization

1. **Configure Geth**: 🏗️ Configure Geth on your computing cluster, using separate terminals for setup and connection.

### Step 3: MPI Environment Setup

1. **Load Modules**: 📦 Load necessary modules and activate the virtual environment for MPI.

   Open two different windows for these commands:
   **Window 1:**
   - `python regionNodeSetup.py 1 1`
   - `python regionNodeSetup.py 1 0`

   **Window 2:**
   - `python regionNodeConnector.py 1`

   ```bash
   module purge
   module load Anaconda3/2020.11
   module load foss/2020a
   source activate $YOURLOCATION/mpienv

### Step 4: Smart Contract Deployment
Proceed with deploying and executing smart contracts as part of your experimental setup.
* run `python submitContract.py`
* run `python smartContract_1.sol`

We hope this guide provides a clear path for setting up and executing your simulated intelligence protocol. Feel free to reach out if you have any questions or need further assistance! 🌟


