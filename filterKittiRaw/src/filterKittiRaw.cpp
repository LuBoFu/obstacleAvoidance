#include <iostream>
#include <filesystem>
#include <h5pp/h5pp.h>
#include <nanotimer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include "options.h"

using namespace std;
using namespace options;
using namespace h5pp;
using namespace pcl;

void cropByDepthLimit(PointCloud<PointXYZI>::Ptr cloud)
{
    options::Options & ops = OptionsInstance::get();
    float distanceLimit = ops.getDouble("distanceLimit", 50);  // in meter
    PassThrough<PointXYZI> filter;
    filter.setInputCloud(cloud);
    filter.setFilterFieldName("z");
    filter.setFilterLimits(0.0, distanceLimit);
    filter.filter(*cloud);
}

void convertToUGVCoordinate(PointCloud<PointXYZI>& cloud)
{
    options::Options & ops = OptionsInstance::get();
    static double cameraHeight = ops.getDouble("cameraHeight", 1.65); // in meter

    for (int i=0; i<cloud.size(); i++){
        cloud[i].y = - cloud[i].y + cameraHeight;
        cloud[i].x *= -1;
    }
}

void cropByHeightRange(PointCloud<PointXYZI>::Ptr cloud)
{
    options::Options & ops = OptionsInstance::get();
    vector<double> limits = ops.getVectorDouble("heightRangeCroppingLimits");
    float groundCroppingThred  = limits.front();
    float ceilingCroppingThred = limits.back();

    PassThrough<PointXYZI> filter;
    filter.setInputCloud(cloud);
    filter.setFilterFieldName("y");
    filter.setFilterLimits(groundCroppingThred, ceilingCroppingThred);
    filter.filter(*cloud);
}

void downSampling(PointCloud<PointXYZI>::Ptr cloud)
{
    options::Options & ops = OptionsInstance::get();
    float leafSize = ops.getDouble("leafSize", 0.5);  // in meter

    VoxelGrid<PointXYZI> downSampler;
    downSampler.setInputCloud(cloud);
    downSampler.setLeafSize (leafSize, leafSize, leafSize);
    downSampler.filter(*cloud);
}

void removeOutliers(PointCloud<PointXYZI>::Ptr cloud)
{
    options::Options & ops = OptionsInstance::get();
    float searchRadius = ops.getDouble("outlierRemovalSearchRadius", 0.5);  // in meter
    int minNeighbors   = ops.getInt("outlierRemovalMinNeighbors", 5);

    RadiusOutlierRemoval<PointXYZI> remover;
    remover.setInputCloud(cloud);
    remover.setRadiusSearch(searchRadius);
    remover.setMinNeighborsInRadius(minNeighbors);
    remover.filter(*cloud);
}

void processFrame(string inputPointCloudFile, string outputPointCloudFile)
{
    options::Options & ops = OptionsInstance::get();

    cout << "to process point cloud file: " << inputPointCloudFile << endl;
    PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);
    PLYReader reader;
    reader.read(inputPointCloudFile, *cloud);
    cout << "original ponit number = " << cloud->size() << endl;

    cropByDepthLimit(cloud);
    cout << "after cropByDistanceLimit(), point number = " << cloud->size() << endl;

    convertToUGVCoordinate(*cloud);

    cropByHeightRange(cloud);
    cout << "after cropByHeightRange(), point number = " << cloud->size() << endl;

    downSampling(cloud);
    cout << "after downSampling(), point number = " << cloud->size() << endl;

    removeOutliers(cloud);
    cout << "after removeOutliers(), point number = " << cloud->size() << endl;

    // write the filtered point cloud.
    PLYWriter plyWriter;
    plyWriter.write(outputPointCloudFile, *cloud, false/*binary*/, false/*use_camera*/);
}

void processSequence(int sequence, File& inputIndex, File& outputIndex)
{
    options::Options & ops = OptionsInstance::get();
    string sequenceDir = string("seq_") + to_string(sequence);

    // copy datasets of the current sequence
    string sourceFilePath = ops.getString("kittiRawIndex");
    outputIndex.copyLinkFromFile( sequenceDir + "/kittiSequenceID",
                      sourceFilePath,  sequenceDir + "/kittiSequenceID");
    outputIndex.copyLinkFromFile( sequenceDir + "/leftImages",
                      sourceFilePath,  sequenceDir + "/leftImages");
    outputIndex.copyLinkFromFile( sequenceDir + "/pointClouds",
                      sourceFilePath,  sequenceDir + "/pointClouds");

    // Read in file names of the point clouds.
    vector<string> inputPointCloudFiles;
    inputIndex.readDataset(inputPointCloudFiles,  sequenceDir + "/pointClouds");
    cout << "frame number: " << inputPointCloudFiles.size() << endl;

    // process the point clouds.
    vector<string> outputPointCloudFiles;
    for (int frame=0; frame<inputPointCloudFiles.size(); frame++){
        if (ops.presents("targetSequence") &&
            ops.presents("numberOfFramesToProcess")){
            if ( frame > ops.getInt("numberOfFramesToProcess", 0) )
                continue;
        }

        string inputPointCloudFile  = inputPointCloudFiles[frame];
        string outputPointCloudFile = ops.getString("kittiRawFilteredDir") + "/" +
                sequenceDir + "/frame_" + to_string(frame) + ".ply";
        using namespace filesystem;
        create_directories( path(outputPointCloudFile).parent_path() );
        outputPointCloudFiles.push_back(outputPointCloudFile);

        processFrame(inputPointCloudFile, outputPointCloudFile);
    }

    // update the output index.
    outputIndex.writeDataset(outputPointCloudFiles, sequenceDir + "/filteredPointClouds",
                             outputPointCloudFiles.size(), H5D_CHUNKED);
}

int main()
{
    OptionsInstance instance("config/filterKittiRawConfig.txt");
    options::Options & ops = OptionsInstance::get();

    File inputIndex (ops.getString("kittiRawIndex"),    FilePermission::READONLY);
    File outputIndex(ops.getString("kittiRawFilteredDir") + "/kittiRawFilteredIndex.h5",
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

