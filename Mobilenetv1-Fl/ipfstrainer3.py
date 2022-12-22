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
print(model_name)

client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5002')  # Connects to: /dns/localhost/tcp/5001/http
res = client.add(model_name)
print(res)

result1 = client.cat(res[0]['Hash'])

#OPTION 1
import hashlib
hash1 = hashlib.sha256(result1).hexdigest()
print(hash1)

hash2 = hashlib.sha256(str(res).hexdigest()
print(hash2)
#hash2 = hashlib.sha256(str(model_name).encode('utf-8').hexdigest()
#print(hash2)
