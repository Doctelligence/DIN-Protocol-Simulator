#from RegionNode import *
from web3 import Web3
from web3.middleware import geth_poa_middleware
from getpass import getpass
import json
import sys
#rn = RegionNode(16)

adjM = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
[1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]]

num_regions = int(sys.argv[1])

adjM = region_data["adj_list"]

#rnos= [10,11,12,13,14,15,16,17,19,20,21,22,23,24,25]
baseNode =  10
enodePairs = {}
for rno in range(num_regions):
    regionNo = rno+baseNode
    w3 = Web3(Web3.IPCProvider("/tmp/region"+str(regionNo)+"_test/geth.ipc"))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    #main_node = w3.admin.nodeInfo
    print("/tmp/region"+str(regionNo)+"_test/geth.ipc")
    for node in range(num_regions):
        #if adjM[rno][node] == 1:
        nbrRegionNo = node+baseNode
        if nbrRegionNo not in enodePairs:
            enodePairs[nbrRegionNo] = []
        enodePairs[nbrRegionNo] = enodePairs[nbrRegionNo] +[w3.geth.admin.node_info()['enode']]

for rno in range(num_regions):
    regionNo = rno+baseNode
    w3 = Web3(Web3.IPCProvider("/tmp/region"+str(regionNo)+"_test/geth.ipc"))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    print(regionNo)
    for node in enodePairs[regionNo]:
        print(node)
        w3.geth.admin.add_peer(node)
