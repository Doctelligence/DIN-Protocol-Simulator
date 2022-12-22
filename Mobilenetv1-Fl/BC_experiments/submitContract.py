from mpi4py import MPI
from regionDriver import *
import numpy as np
import subprocess
import sys
import json
import random
import time
from multiprocessing import Pool,Array
from regionDriver import *
import numpy as np
from web3 import Web3
from web3.middleware import geth_poa_middleware

import solc
from solc import compile_source, compile_files, link_code
from web3 import Web3
from web3.middleware import geth_poa_middleware

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

def w3setup(region):
    w3 = Web3(Web3.IPCProvider("/tmp/region"+str(region)+"_test/geth.ipc"))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    # set pre-funded account as sender
    w3.eth.defaultAccount = w3.eth.accounts[0]
    w3.geth.personal.unlockAccount(w3.eth.accounts[0],"password",0)
    return w3

def registerContract(region,num_regions):
        w3 = w3setup(region)
        source = ""
        lines_list = []

        #print "detectGlobalAttack_"+str(num_regions)+".sol"
        with open("smartContract_"+str(num_regions)+".sol","r") as contract:
            lines_list = contract.readlines() #[source+l for l in ]

        for l in lines_list:
            source = source +l

        # Basic contract compiling process.
        #   Requires that the creating account be unlocked.
        #   Note that by default, the account will only be unlocked for 5 minutes (300s).
        #   Specify a different duration in the geth personal.unlockAccount('acct','passwd',300) call, or 0 for no limit

        compiled = compile_source(source)

        contract_interface = compiled['<stdin>:GlobalAttackDetector']


        #Instantiate and deploy contract
        print("Generating Contract")
        Greeter = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])

        # Submit the transaction that deploys the contract
        print("Submitting contract")
        tx_hash = Greeter.constructor().transact()

        # # # Wait for the transaction to be mined, and get the transaction receipt
        print("Waiting for Contract to be mined")
        tx_receipt = w3.eth.waitForTransactionReceipt(tx_hash,1800)
        contractAddr = tx_receipt.contractAddress
        print("Contract Mined!")
        contractData = {}
        contractData['contractAddr'] = contractAddr
        contractData['contract_interface'] = contract_interface['abi']
        with open('contractDetails.out', 'w') as outfile:
            json.dump(contractData, outfile)

registerContract(10,1)
