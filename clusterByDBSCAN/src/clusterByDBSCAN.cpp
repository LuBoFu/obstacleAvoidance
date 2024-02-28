#include <iostream>
#include <filesystem>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <h5pp/h5pp.h>
#include <nanotimer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <mlpack.hpp>
#include <armadillo>
#include "options.h"

using namespace std;
using namespace options;
using namespace h5pp;
using namespace pcl;
using namespace mlpack;

void generateKDistData(const PointCloud<PointXYZI>& cloud, string dataFilename)
{
    // Construct the reference dataset.
    arma::mat dataset(3, cloud.size());
    for (size_t c=0; c<dataset.n_cols; c++){
        dataset(0, c) = cloud[c].x;
        dataset(1, c) = cloud[c].y;
        dataset(2, c) = cloud[c].z;
    }

    // knn search
    typedef NeighborSearch<NearestNeighborSort, EuclideanDistance,
                           arma::mat, RTree> NeighborSearchType;
    NeighborSearchType knn(dataset);
    arma::Mat<size_t> neighbors;
    arma::mat distances;
    const int k=6;
    knn.Search(k, neighbors, distances);

    // we are only interested with the k-dist(the last row)
    arma::mat sortedDistances = arma::sort(distances.tail_rows(1),
                                           "descend", 1/*along the row*/);
    sortedDistances.save(dataFilename, arma::csv_ascii);
}

void cluster(const PointCloud<PointXYZI>& cloud, string frameDir)
{
    options::Options & ops = OptionsInstance::get();

    // Construct the reference dataset.
    arma::mat dataset(3, cloud.size());
    for (size_t c=0; c<dataset.n_cols; c++){
        dataset(0, c) = cloud[c].x;
        dataset(1, c) = cloud[c].y;
        dataset(2, c) = cloud[c].z;
    }

    // construct the clusterer.
    double epsilon = ops.getDouble("Eps", 1.0);
    int minPoints  = ops.getInt("MinPts", 6);
    DBSCAN<> dbscan(epsilon, minPoints);

    // perform the clustering.
    arma::Row<size_t> assignments;
    arma::mat centroids;
    int clusterNumber = dbscan.Cluster(dataset, assignments, centroids);
    cout << dataset.size() << " points have been clustered into "
         << clusterNumber << " clusters" << endl;

    // save the clustering result.
    assignments.save( frameDir + "/assignments.csv", arma::csv_ascii);
    centroids.save  ( frameDir + "/centroids.csv",   arma::csv_ascii);

    // render the point cloud so that different clusters have visually distinguishable colors.
    PointCloud<PointXYZI> renderedCloud(cloud);
    for (int i=0; i<renderedCloud.size(); i++){
        if (assignments[i] == SIZE_MAX){
            // A noise point
            renderedCloud[i].intensity = 0;
        }else{
            // A cluster point. shuffle its cluster ID.
            static int flag = 0;
            static vector<int> mappingTable(255);
            if (flag == 0){
                flag = 1;
                for (int i=0; i<255; i++){
                    mappingTable[i] = i;
                }
                std::shuffle(mappingTable.begin(), mappingTable.end(),
                             default_random_engine() );
            }
            size_t shuffledClusterID = mappingTable[assignments[i] % 255];
            renderedCloud[i].intensity = 1 + shuffledClusterID;
        }
    }

    PLYWriter plyWriter;
    plyWriter.write(frameDir + "/pointCloudWithClusterInfo.ply",
                    renderedCloud, false/*binary*/, false/*use_camera*/);
}

void processFrame(string inputPointCloudFile, string frameDir)
{
    options::Options & ops = OptionsInstance::get();

    cout << "to process point cloud file: " << inputPointCloudFile << endl;
    PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);
    PLYReader reader;
    reader.read(inputPointCloudFile, *cloud);
    cout << "ponit number = " << cloud->size() << endl;

    if (ops.presents("onlyGenerateKDistData")){
        string dataFilename = frameDir + ".csv";
        generateKDistData(*cloud, dataFilename);
        return;
    }

    cluster(*cloud, frameDir);
}

void processSequence(int sequence, File& inputIndex, File& outputIndex)
{
    options::Options & ops = OptionsInstance::get();
    string sequenceDir = string("seq_") + to_string(sequence);

    // Read in file names of the point clouds.
    vector<string> inputPointCloudFiles;
    inputIndex.readDataset(inputPointCloudFiles,  sequenceDir + "/filteredPointClouds");
    cout << "frame number: " << inputPointCloudFiles.size() << endl;

    // process the point clouds.    
    for (int frame=0; frame<inputPointCloudFiles.size(); frame++){
        if (ops.presents("targetSequence") &&
            ops.presents("numberOfFramesToProcess")){
            if ( frame >= ops.getInt("numberOfFramesToProcess", 0) )
                continue;
        }

        string inputPointCloudFile  = inputPointCloudFiles[frame];
        string frameDir = ops.getString("clustersDir") + "/" +
                          sequenceDir + "/frame_" + to_string(frame);
        filesystem::create_directories(frameDir);

        processFrame(inputPointCloudFile, frameDir);
    }
}

int main()
{
    OptionsInstance instance("config/clusterByDBSCANConfig.txt");
    options::Options & ops = OptionsInstance::get();

    File inputIndex(ops.getString("kittiRawFilteredDir") + "/kittiRawFilteredIndex.h5",
                    FilePermission::READONLY);
    filesystem::create_directories(ops.getString("clustersDir"));
    File outputIndex(ops.getString("clustersDir") + "/clustersIndex.h5",
                     FilePermission::REPLACE);

    int sequenceNumber;
    inputIndex.readDataset(sequenceNumber, "/sequenceNumber");
    cout << "to process " << sequenceNumber << " sequences" << endl;
    outputIndex.writeDataset(sequenceNumber, "/sequenceNumber");

    for (int i=0; i<sequenceNumber; i++){
        if (ops.presents("targetSequence")){
            if ( i != ops.getInt("targetSequence", 0) )
                continue;
        }

        processSequence(i, inputIndex, outputIndex);        
    }
    return 0;
}

