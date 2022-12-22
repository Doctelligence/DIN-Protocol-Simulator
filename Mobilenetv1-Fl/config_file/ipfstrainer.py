from ConfigReader import ConfigReader
from Learner import Learner
import ipfshttpclient
import numpy as np
#import sha256

patient_id = 1
num_patients = 1

cr = ConfigReader()
learner = Learner(patient_id,num_patients)

model_name = learner.saveModel(1)

client = ipfshttpclient.connect()  # Connects to: /dns/localhost/tcp/5001/http
res = client.add(model_name)


#result1 = client.cat(res["Hash"])

#OPTION1
#write result1 to a file file1.

#compare file1 to the file denoted by model_name

#OPTION2
#get sha256(result1) = hash1

#compare res["Hash"] with hash1

