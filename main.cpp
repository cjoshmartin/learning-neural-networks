#include <iostream>
#include <vector>
#include "Net.h"


int main() {

    // e.g. { 3, 2, 1 }
    std::vector<unsigned> topology;

    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);

    Net myNet(topology);

    std::vector<double> inputVals;
    myNet.feedForward(inputVals); // tell the nextwork  its inputs

    const std::vector<double> targetVals;
    myNet.backProp(targetVals); // telling the neural network its expected outputs

    std::vector<double> resultVals;
    myNet.getResults(resultVals); // get the results from the neural network after it is done trainning

    return 0;
}