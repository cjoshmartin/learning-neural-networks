//
// Honor Pledge:
//
// I pledge that I have neither given nor
//  received any help on this assignment.
//
//chajmart

//

#ifndef LEARNING_CNN_TRAININGDATA_H
#define LEARNING_CNN_TRAININGDATA_H


#include <string>
#include <fstream>

class TrainingData
{
public:
    TrainingData(const std::string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(std::vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(std::vector<double> &inputVals);
    unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

private:
    std::ifstream m_trainingDataFile;
};


#endif //LEARNING_CNN_TRAININGDATA_H
