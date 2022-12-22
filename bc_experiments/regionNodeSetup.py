import sys, os, time
import json
import numpy as np
import subprocess
#from solc import compile_source, compile_files, link_code
from web3 import Web3
from web3.middleware import geth_poa_middleware

import re

totalRegions = int(sys.argv[1])
setupRegion = int(sys.argv[2])

if setupRegion == 1:
	with open('initTemp.json') as json_file:
		initJson = json.load(json_file)
	initJson["alloc"] = {}
	with open('initTemp.json', 'w') as outfile:
		json.dump(initJson, outfile)
	for rno in range(totalRegions):
		regionNo = rno+10
		#delete existing regions
		runStr = "rm -rf ./region"+str(regionNo)+"_test"
		runStrList = runStr.split(" ")
		rawResult = subprocess.run(runStrList)

		#Step 1: Create an account
		#geth --datadir ./region1_test account new --password ./pwd.txt
		runStr = "geth --nousb --datadir ./region"+str(regionNo)+"_test account new --password ./pwd.txt"
		print(runStr)
		runStrList = runStr.split(" ")
		rawResult = subprocess.run(runStrList, stdout=subprocess.PIPE)
		add = rawResult.stdout.split(b':')[1].splitlines()[0].decode('utf-8')
		#print()
		accountNo = add.replace(" ","")#add[add.find("Public address of the key:	")+1:]
		print(accountNo)

		#Step 2: Populate the account with test ether
		initJson = {}
		with open('initTemp.json') as json_file:
			initJson = json.load(json_file)
		initJson["alloc"][str(accountNo)] = {"balance":"11111111111111111111111111111111"}
		#print()
		with open('initTemp.json', 'w') as outfile:
			json.dump(initJson, outfile)

	for rno in range(totalRegions):
		regionNo = rno+10
		runStr = "geth --datadir ./region"+str(regionNo)+"_test init initTemp.json 2> /tmp/logDump"+str(regionNo)
		#runStrList = runStr.split(" ")
		#f = open("/tmp/logDump"+str(regionNo),"w")
		rawResult = subprocess.Popen(runStr,shell=True)
		#f.close()

	#Step 3: Run the miner on each
if setupRegion == 0:
	for rno in range(totalRegions):
		regionNo = rno+10
		rpcport = 8545+rno
		port = 30303+rno

		#runStr = "ls -lt"
		runStr = "geth --allow-insecure-unlock --nodiscover --rpc --rpcaddr 127.0.0.1 --rpccorsdomain \"*\" --rpcapi \"eth,net,web3,miner,debug,personal,rpc\" --mine --minerthreads 1 --rpcport "+str(rpcport)+" --port "+str(port)+" --datadir ./region"+str(regionNo)+"_test --networkid 2018 2> /tmp/logDump"+str(regionNo)
		runStrList = runStr.split(" ")
		print(runStr)
		subprocess.Popen(runStr, shell=True)

# for rno in range(1):
#	  regionNo = rno+18
#	  rpcport = 8546+rno
#	  port = 303+rno
#
#	  #runStr = "ls -lt"
#	  runStr = "geth --rpc --rpcaddr 127.0.0.1 --rpccorsdomain \"*\" --rpcapi \"eth,net,web3,miner,debug,personal,rpc\" --mine --minerthreads 1 --rpcport "+str(rpcport)+" --port "+str(port)+" --datadir ./region"+str(regionNo)+"_test --networkid 2018 2>> /tmp/logDump"+str(regionNo)
#	  runStrList = runStr.split(" ")
#	  print(runStr)
#	  subprocess.Popen(runStr, shell=True)
