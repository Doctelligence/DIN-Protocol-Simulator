## Simulated FL PHR Environment (for Classification tasks)

Download a healthcare dataset e.g.: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

# Step 1:
a)Activate Virtual Python Learning Environment
b)Create Simulated PHRs for Patients: In the script 'data-partition.py' set the 'no_patients' variable to the number of patients you'd like to use in your experiment.
-Set the location for the for the patient directories to be saved.
-The script will attempt to equally divide the number of negative and positive classes to each patient.

c)Run `python data-partition.py`

# Step 2:
Go to 'train.py' script.
a)Import a TensorflowLite Model (e.g. a MobileNet version).
- Search for different model-types at https://tfhub.dev/
-Specify variable 'module_selection' with model variables,e.g. name,input size, etc)
-Specify MODULE_HANDLE

b)Set FL-Variables in the 'train.py' script:
1.Batch_size =
2.num_epochs =
3.max_rounds =
4.no_patients =

# Step 3:
a)Create a 'Global_model' subdirectory in a specified location to save the global_models output
- The Model will save in a sub-directory with the name of the FL-variables so that you can identify which model it was that was trained.
b)Run 'python train.py' (advisable - advanced computing cluster)


## Run Distributed Version
# Step 1: 
Run `export FL_CONFIG_FILE=/path/to/config/file`
In config file the following parameters are specified

* `raw_data_root_dir` : this is the place where the raw chest xray data set is stored prior to partition

* `experiment_id`: put any string without spaces over here to uniquely identify your experiment. could be anything as long as you can uniquely identify it
examples names: 
    1. num_patients and num_rounds, like "10_patients_5_rounds"
    2. data time of creation, like "17th_Oct_2021_17-56-09-UTC"
    3. unix timestamp of creation, like "1639531877"
    
* `directories`: the following are directory names. 
  1. `root_dir`: the root directory where chestxray train, test and val folders are located
  2. `data_root_dir`: patient training data location name. for eg. if experiment_id = 1639531877, and root_dir: "/tmp/federated_learning_dataset/", then based on the below names data_root_directory is "/tmp/federated_learning_dataset/1639531877/data_root_dir/"
  3. `test_dir`: test data location name
  4. `val_dir`: validation data location name
  5. `model_save_dir`: model save directory name

# Step 2:
Run data partitioner `python data-partition.py` that takes the config file from `export FL_CONFIG_FILE=/path/to/config/file`

# Step 3:
Make sure mpi4py is installed. Refer to the following [link](https://www.arc.ox.ac.uk/using-python-mpi-arc)

# Step 4:
Use MPI to run
`mpirun -n <no_patients> python DistributedTrainer.py`

## Run IPFS Private Node
# Step 1:
* Setup private ipfs node: https://labs.eleks.com/2019/03/ipfs-network-data-replication.html 
* Run `ipfs daemon`

# Step 2
* Export config file as above in `## Run Distributed Version`

## Step 3
Run `python ipfstrainer.py`

## Setup Private Blockchain to work with Computing Cluster
# Step 1
* Install dependencies, geth, etc.
* Your IT cluster controller may have to install these for you

# Step 2
Run `geth/go:
srun -p interactive --pty /bin/bash
module spider geth --> geth: geth/2022
module load geth/2022` on your computing cluster
* OR `module load geth` 

# Step 3
Open two different windows for these commands.
Window 1
* `python regionNodeSetup.py 1 1`
* `python regionNodeSetup.py 1 0`
Window 2
* `python regionNodeConnector.py 1`

# Step 4 
Load mpi:
module purge
module load Anaconda3/2020.11
module load foss/2020a
source activate $YOURLOCATION/mpienv

# Step 5
* run `python submitContract.py`
* run `python smartContract_1.sol`

