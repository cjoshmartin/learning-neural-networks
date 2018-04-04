//
// Honor Pledge:
//
// I pledge that I have neither given nor
//  received any help on this assignment.
//
//chajmart

//

#ifndef LEARNING_CNN_NET_H
#define LEARNING_CNN_NET_H

#include <vector>

#include "Neuron.h"
class Neuron; // forward def.
typedef std::vector<Neuron> Layer;

class Net {

public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;

private:
    std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageSmoothingFactor;
    double m_recentAverageError;
};


#endif //LEARNING_CNN_NET_H
