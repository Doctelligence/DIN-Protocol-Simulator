from ConfigReader import ConfigReader
from Learner import Learner
import ipfshttpclient
import numpy as np
import hashlib

patient_id = 1 
num_patients = 1

#cr = ConfigReader()
learner = Learner(patient_id, num_patients)

model_name = learner.saveModel(1)
print(model_name)

with open(model_name,"rb") as f:
    bytes = f.read() # read entire file as bytes
    readable_hash = hashlib.sha256(bytes).hexdigest();
    print(readable_hash)

client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5002') # Connects to:dns/localhost/tcp/5001/http
res = client.add(model_name)

result1 = client.cat(res['Hash'])
#print(result1)

hash1 = hashlib.sha256(result1).hexdigest()
print(hash1)

#hash2 = hashlib.sha256(learner.model).hexdigest()
#print(hash2)
