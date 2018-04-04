//
// Honor Pledge:
//
// I pledge that I have neither given nor
//  received any help on this assignment.
//
//chajmart

//

#include <iostream>
#include <cassert>
#include <cmath>
#include "Net.h"

Net::Net(const std::vector<unsigned> &topology) {

    unsigned numLayers = topology.size();

    // creates a layer
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = (layerNum == numLayers ) - 1 ? 0 /*hidden layer */ :  topology[layerNum +1] /* Output layer */; // num of lays depends on if it is a
                                                                                        // hidden layer or output layer

        // We have made a new Layer in our CNN, now we have to fill it with neurons
        // and add a bias neuron to each layer
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs,neuronNum));
            std::cout << "Made a Neuron\n";
        }
    }


    // Forward Propagation
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum-1];

        for (int n = 0; n <m_layers[layerNum].size() - 1 ; ++n)
        {
            m_layers[layerNum][n].feedForward(prevLayer); // tells each neuron in each to layer to get busy with the
                                                // math.
        }
        // force the bias node's output value to 1.0. its th last neuron created above
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const std::vector<double> &inputVals) {

    assert(inputVals.size() == m_layers[0].size() - 1);

    for(unsigned i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
}

void Net::backProp(const std::vector<double> &targetVals) {
    // this functions will need to:
        // * Calculate overall net error of the Neural Net (RMS or Root Mean Square Error, of the output neuron errors)
        // * Calculate output layer gradients
        // * Calculate gradients on hidden layers
        // * for all layers from outputs to first hidden layer, update connect weights

    // Calculate overall net error of the Neural Net (RMS or Root Mean Square Error, of the output neuron errors)
    // this is what this algorithm is trying to mimify
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    for (unsigned n = 0; n < outputLayer.size() -1 ; ++n)
    {

        double delta = targetVals[n] - outputLayer[n].getOutputVal(); // RMS
        m_error+= delta * delta; // RMS
    }
    m_error /= outputLayer.size() - 1 ; // RMS
    m_error = sqrt(m_error);// RMS

    // Implement a recent average measurement
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) /  (m_recentAverageSmoothingFactor + 1.0);

    //Calculate output layer gradients (linear regresstion)
    for (unsigned n = 0; n <outputLayer.size() - 1 ; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    
    // Calculate gradients on hidden layers
    for (unsigned layerNum = m_layers.size(); layerNum > 0 ; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        // each neuron
        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // for all layers from outputs to first hidden layer, update connect weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        //each neuron
        for (unsigned n = 0; n < layer.size() -1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(std::vector<double> &resultVals) const {
   resultVals.clear();
    for (int n = 0; n < m_layers.back().size() -1 ; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}
