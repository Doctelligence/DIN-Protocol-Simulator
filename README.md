## DIN Protocol Simulator

### DIN-Protocol Simulator Overview

The **DIN-Protocol Simulator** begins with **smart contracts** running on a **test environment** powered by a **traditional Ethereum network** (Solidity). These smart contracts automate the coordination of the training process, track contributions, and distribute rewards to participants. This decentralized framework ensures privacy, security, and fairness in the training of AI models.

The system interacts with **IPFS** (InterPlanetary File System) for decentralized model storage, ensuring that AI models are securely stored and shared across participants while maintaining data integrity. This interaction allows the models to remain decentralized, ensuring that participants retain control of their data and contribute to the collective training process.

Designed for **advanced computing clusters**, the simulator provides a **test environment** that enables experimentation in a **reproducible, scalable** way. Researchers can use this environment to experiment with federated learning and explore the potential of decentralized AI in traditional research settings.

This simulator lays the foundational groundwork for our **[live demo prototype](https://github.com/Doctelligence/DIN-Prototype)**, where we demonstrate the system's ability to evaluate model contributions and issue rewards based on those contributions in a **fully decentralized manner**. In the prototype, smart contracts handle the distribution of rewards, providing participants with incentives for their contributions in a transparent, automated way, based on the value they bring to the training process.

The simulator uses **healthcare data** as a demonstration use case, showing how personal health records can remain under individual control while still contributing to the collective training of an AI model. Successful training of AI models to a high **AUC (Area Under the Curve)** in this environment demonstrates the potential of federated learning for decentralized, privacy-preserving applications.

**Key Highlights:**
- **Smart contracts** coordinate the training process, track contributions, and distribute rewards in a **test environment** using a **traditional Ethereum network**.
- Interacts with **IPFS** for decentralized model storage, ensuring privacy, security, and data integrity.
- Provides a **test environment** for experimentation in **advanced computing clusters**, supporting reproducibility and scalability.
- Lays the groundwork for the **[live demo prototype](https://github.com/Doctelligence/DIN-Prototype)**, which demonstrates the ability to evaluate model contributions and issue rewards in a **fully decentralized** way.
- Uses **healthcare data** as a use case to demonstrate **Federated Learning (FL)** in a privacy-preserving, decentralized setting.
- Successfully trained AI models to a high **AUC**, validating the effectiveness of decentralized AI in real-world applications.

**Try it yourself!** Explore the **DIN-Protocol Simulator** and see how decentralized AI, smart contracts, and data ownership can be applied to real-world, privacy-preserving scenarios, setting the stage for a fully decentralized reward distribution system in our **[live demo prototype](https://github.com/Doctelligence/DIN-Prototype)**.

------------------------------------------

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


