import os
import glob
from ConfigReader import ConfigReader
import shutil

cr = ConfigReader()

#Data Path
#data_path = '/experiments/datasets/Chest_xray/train/'
data_path = cr.raw_data_root_dir

exp_id = cr.experiment_id
root_dir = cr.root_dir

data_root_dir = cr.data_root_dir
val_dir = cr.val_dir
test_dir = cr.test_dir
model_save_dir = cr.model_save_dir

#get Number of Patients for Your Experiment
no_patients = cr.no_patients

if not os.path.exists(root_dir):
    os.mkdir(root_dir)
    print("root directory {} created".format(root_dir))
else:
    print("the data for experiment_id {} already exists....quitting!".format(exp_id))
    exit(1)

if not os.path.exists(data_root_dir):
    os.mkdir(data_root_dir)
    print("data root directory {} created".format(data_root_dir))
else:
    print("data root directory {} exists".format(data_root_dir))

if not os.path.exists(val_dir):
    os.mkdir(val_dir)
    print("val directory {} created".format(val_dir))
else:
    print("val directory {} exists".format(val_dir))

if not os.path.exists(test_dir):
    os.mkdir(test_dir)
    print("test directory {} created".format(test_dir))
else:
    print("test directory {} exists".format(test_dir))

if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
    print("model_save directory {} created".format(model_save_dir))
else:
    print("model_save directory {} exists".format(model_save_dir))

files = os.listdir(data_path)
print("Files and directories:")
print(data_path)

#Sanity Check
if not os.path.isdir(data_path):
    print('Is not a directory ', data_path)

files_list = []
for root, directories, files in os.walk(data_path):
    for name in files:
        files_list.append(os.path.join(root, name))

#Classification Task/Label Groups
files_list_pos = glob.glob(data_path+"/"+"PNEUMONIA/*")
print(files_list_pos)

files_list_neg = glob.glob(data_path+"/"+"NORMAL/*")
print(files_list_neg)

#Divide Data into Labelled Groups for Each Patient
NORMAL  = int(len(files_list_pos)/no_patients)
PNEUMONIA = int(len(files_list_neg)/no_patients)

if NORMAL  ==0:
    print('Normal is < 1')
    exit(0)
if PNEUMONIA ==0:
    print('Pneumonia is < 1')
    exit(0)

Patient_list = data_root_dir
print(Patient_list)
for i in range(no_patients):
    os.mkdir(Patient_list + '/Patient_' + str(i))
    os.mkdir(Patient_list + '/Patient_' + str(i) + '/NORMAL')
    os.mkdir(Patient_list + '/Patient_' + str(i) + '/PNEUMONIA')

#Deposit Data into Simulated PHR Databases


i = 0
num_file = 0

for file in files_list_neg:
    shutil.copy(file, Patient_list + '/Patient_' + str(i) + '/NORMAL/')
    num_file = num_file + 1
    if num_file == NORMAL:
        num_file = 0
        i = i + 1
        if i == (no_patients):
            break

i = 0
num_file = 0

for file in files_list_pos:
    shutil.copy(file, Patient_list + '/Patient_' + str(i) + '/PNEUMONIA/')
    num_file = num_file + 1
    if num_file == PNEUMONIA:
        num_file = 0
        i = i + 1
        if i == (no_patients):
            break
