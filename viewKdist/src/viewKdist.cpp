#include <iostream>
#include <filesystem>
#include <boost/algorithm/string.hpp>
#include <h5pp/h5pp.h>
#include <matplot/matplot.h>
#include <armadillo>
#include "options.h"

using namespace std;
using namespace options;
using namespace h5pp;

const int xDivisions = 2000;
const int yDivisions = 1000;
const double yMax = 12;   // in meters.

void processSequence(int sequence, File& inputIndex, arma::mat& m)
{
    options::Options & ops = OptionsInstance::get();
    string sequenceDir = string("seq_") + to_string(sequence);

    // Determine number of frames
    vector<string> pointCloudFiles;
    inputIndex.readDataset(pointCloudFiles,  sequenceDir + "/filteredPointClouds");
    int frameNumber = pointCloudFiles.size();
    cout << "to process sequence " << sequence << endl;

    #pragma omp parallel
    {
        #pragma omp for
        for (int frame=0; frame<frameNumber; frame++){
            string csvFile = ops.getString("clustersDir") + "/" +
                             sequenceDir + "/frame_" + to_string(frame) + ".csv";
            arma::rowvec v;
            v.load(csvFile, arma::csv_ascii);

            // update the counters
            for (int i=0; i<v.size(); i++){
                int x =  i * 1.0 / v.size() * m.n_cols;
                double regulatedValue = ( v(i)>= yMax? yMax - 1e-4: v(i) );
                int y =  m.n_rows - 1 - regulatedValue / yMax * m.n_rows;
                #pragma omp critical
                {
                    m(y, x)++;
                }
            }

            cout << ".";
        }
    }
    cout << endl;
}

int main(int argc, const char* argv[])
{
    OptionsInstance instance(argc, argv);
    options::Options & ops = OptionsInstance::get();

    File inputIndex( ops.getString("clustersDir") + "/clustersIndex.h5",
                     FilePermission::READONLY );
    int sequenceNumber;
    inputIndex.readDataset(sequenceNumber, "/sequenceNumber");    

    arma::mat m(yDivisions, xDivisions, arma::fill::zeros);
    for (int i=0; i<sequenceNumber; i++){        
        processSequence(i, inputIndex, m);
        //if (i==4) break;
    }
    m.save("accumulatedKdist.csv", arma::csv_ascii);

    using namespace matplot;
    vector<vector<double>> data(m.n_rows);
    for (int r=0; r<data.size(); r++){
        data[r].resize( m.n_cols);
        for (int c=0; c<data[r].size();c++){
            data[r][c] = log( m(r,c) + 1 );
        }
    }

    auto fig = gcf(true);
    imagesc(data);
    colorbar();
    fig->save("accumulatedKdist.png");
    fig->show();
    return 0;
}

