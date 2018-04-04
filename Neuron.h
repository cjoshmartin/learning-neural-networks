//
// Honor Pledge:
//
// I pledge that I have neither given nor
//  received any help on this assignment.
//
//chajmart

//

#ifndef LEARNING_CNN_NEURON_H
#define LEARNING_CNN_NEURON_H

#include <vector>
#include <cstdlib>
#include "Net.h"


struct Connection
{
    double weight;
    double deltaWeight;
};


class Neuron; // forward def.
typedef std::vector<Neuron> Layer;

class Neuron {

public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void feedForward(const Layer &prevLayer);

    void setOutputVal(double m_outputVal);

    double getOutputVal() const;

    void calcOutputGradients(double targetVal);

    void calcHiddenGradients(const Layer &nextLayer);

    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;  // [0.0,1.0] overall net training rate
    static double alpha ; // [0.0,n] multiplier of the last weight change (momentum)
    static double randomWeight(void){ return rand()/ double(RAND_MAX); } // gives me a number between [0,1]
    double m_outputVal; // output of each neuron :)
    std::vector<Connection> m_outputWeights; // weigths of each Neuron
    unsigned m_myIndex;
    double m_gradient;
    double transferFunction(double sum);
    double transferFunctionDerivative(double sum);
    double sumDOW(const Layer &nextLayer) const;
};


#endif //LEARNING_CNN_NEURON_H
