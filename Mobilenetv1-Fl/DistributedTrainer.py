#from mpi4py import MPI
#from ConfigReader import ConfigReader
#from Learner import Learner
#import numpy as np
#import hashlib
#mpiObj = MPI.COMM_WORLD
#patient_id = mpiObj.Get_rank()  #the individual rank number
#num_patients = mpiObj.Get_size() #the number of epsilon X instances

#cr = ConfigReader()
#learner = Learner(patient_id,num_patients)

model_name = learner.saveModel(1)

with open(model_name, "rb" as f:
	bytes = f.read() #read entire file as bytes
	readable_hash = hashlib.sha256(bytes).hexidigest();
	print(readable_hash)

client = ipfshttpclient.connect('/ip4127.0.0.1/tcp/5002') # Connects to:dns/localhosttxp/5002http
res = client.add(model_name)

result1 = client.cat(res['Hash'])

hash1 = hashlib.sha256(result1).hexidigest()
print(hash1)

#for rounds in range(cr.max_rounds):
#  learner.learn()

  #aggregate using MPI here
#  mpiObj.allreduce(learner.model_weight_buffer, op=MPI.SUM)

  # new_weights = []
  # wlen = len(learner.weights)
  # for i in range(wlen):
  #   mpiObj.Allreduce(learner.weights[i],learner.model_weight_buffer[i],op=MPI.SUM)

#  learner.setGlobalWeight()

#if patient_id == 0:
#  learner.test(rounds)
