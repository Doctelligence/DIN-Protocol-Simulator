pragma solidity >=0.4.22 <0.6.0;
contract GlobalSmartContract {

    address  owner;

    uint numRegions;

    constructor() public {
        owner = msg.sender;
        numRegions = 4;
    }
}
