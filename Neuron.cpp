//
// Honor Pledge:
//
// I pledge that I have neither given nor
//  received any help on this assignment.
//
//chajmart

//

#include <cmath>
#include "Neuron.h"

double Neuron::eta = .15; // overall net learning rate [0.0,1.0]
double Neuron::alpha = .5; // momentum, multiplier of last deltaWeight [0.0, n]


Neuron::Neuron(unsigned numOutputs, unsigned myIndex) : m_myIndex(myIndex) {
     for(unsigned connections = 0; connections < numOutputs; ++connections)
     {
          m_outputWeights.push_back(Connection());
          m_outputWeights.back().weight = randomWeight(); // talking to the neuron we just created and
                                                         // giving it a random weight
     }
}

void Neuron::setOutputVal(double m_outputVal) {
    Neuron::m_outputVal = m_outputVal;
}

double Neuron::getOutputVal() const {
    return m_outputVal;
}

void Neuron::feedForward(const Layer &prevLayer) {

    double sum = 0.0;

    // sum the previous layer's outputs (which are our inputs)
    // include the bias node from the previous layer

    for (unsigned n = 0; n < prevLayer.size() ; ++n) {

       sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = Neuron::transferFunction(sum); // also classed the activation function
}

double Neuron::transferFunction(double x) {
    // The brain of the whole thing.
    // think of ece301 when you look at this function
    // meaning a transfer function curve like e^x is ideal
    // but a transfer function can also be a unit step function or a ramp or an impulse function

    // we are going to choose to use tanh(x) <==> (e^x - e^(-x))/ (e^x + e^(-x))
        // NOTE: this transfer function range is [-1.0,1.0]


    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
    // The brain of the whole thing.
    // think of ece301 when you look at this function

    // derivative of tanh
    // or 1 - tanh^2 (x)

    return 1.0 - x * x; // or this
}

void Neuron::calcOutputGradients(double targetVal) {

    double  delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunction(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {

    double dow = sumDOW(nextLayer); // sum of the weights of the next layer
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const {

    double  sum = 0.0;

    // Sum our contribution of the errors at the nodes we feed

    for (int n = 0; n < nextLayer.size() - 1 ; ++n)
    {
        sum += m_outputWeights[n].weight *nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer) {

    // The weight to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (int n = 0; n < prevLayer.size() ; ++n) {

        Neuron &neuron = prevLayer[n];

        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double  newDeltaWeight =
                // Individual input, magnified by the gradient and train rate
                    eta // overall learning rate
                    * neuron.getOutputVal()
                    * m_gradient
                    * alpha // Also add momentum = a fraction of the previous delta weight
                    * oldDeltaWeight // save state
                    ;

    }

}

