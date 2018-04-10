#include <iostream>
#include <vector>
#include <assert.h>
#include "Net.h"
#include "TrainingData.h"

#include "string_formater.h"

using namespace std;

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

void genrateData(std::string store_path) {
    // Radnom training sets of XOR -- two inputs and one output

    std::string outputStr;

    outputStr += "topology: 2 4 1\n";

    for (int i = 2000; i >=0; --i) {
        int n1 = (int)(2.0 * rand() / double(RAND_MAX));
        int n2 = (int)(2.0 * rand() /double(RAND_MAX));
        int t = n1 ^ n2; // should be 0 or 1
        outputStr += string_formater::formater( "in: %d.0 %d.0 \nout: %d.0\n", n1,n2,t);
    }

    std::ofstream myfile;
    myfile.open (store_path);
    myfile << outputStr;
    myfile.close();

}

int main() {
    genrateData("trainingData.txt");

    TrainingData trainData("trainingData.txt");

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
             << myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;
    return 0;
}