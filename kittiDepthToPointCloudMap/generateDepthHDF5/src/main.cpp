#include <iostream>
#include <fstream>
#include <set>
#include <boost/algorithm/string.hpp>
#include <h5pp/h5pp.h>
#include <filesystem>
#include <Eigen/Core>
#include "options.h"
#include "fileIO.h"

using namespace std;
using namespace h5pp;
using namespace options;

void deleteSomeDepthSequences(vector<string>& depthSequencePaths)
{
    set<string> sequenceIDSet = {
        "2011_09_28_drive_0016_sync",
        "2011_09_28_drive_0021_sync"
    };

    vector<string>& paths = depthSequencePaths;
    for (auto it= paths.begin(); it!=paths.end(); ){
        filesystem::path p(*it);
        if ( sequenceIDSet.count(p.filename().string()) == 1 ){
             it = paths.erase(it);
             continue;
        }
        it++;
    }
}

// read the P_rect_02 line into a matrix.
void loadLeftCameraProjectMatrix(string filename, Eigen::MatrixXf& m)
{
    ifstream inputFile(filename);
    if (!inputFile){
        cout << "could not open " << filename << endl;
        exit(-1);
    }
    string line;
    while (getline(inputFile,line)){
        if (!boost::starts_with(line, "P_rect_02:")) continue;
        istringstream inputStream(line);
        inputStream.ignore( string("P_rect_02:").length() );
        for (int i=0; i<m.rows(); i++){
            for (int j=0; j<m.cols(); j++){
                inputStream >> m(i,j);
            }
        }
    }
//    cout << "from " << filename << ", read matrix: " << endl
//         << m << endl;
}


void processDepthSequence(string depthSequencePath, string sequenceH5Dir,
                          const vector<string>& rawSequencePaths, File& outputFile)
{
    using namespace filesystem;
    cout << "processing " << depthSequencePath << endl;
    // get sequence kitti ID
    // sequencePath looks like: /sw/cv/dbase/kittiDepth/train/2011_09_26_drive_0001_sync
    filesystem::path p(depthSequencePath);
    string kittiSequenceID = p.filename().string();
    outputFile.writeDataset( kittiSequenceID, sequenceH5Dir + "/kittiSequenceID");

    // construct the full path of the depth seqence
    filesystem::path  fullDepthSequencePath(depthSequencePath +
                                            "/proj_depth/groundtruth/image_02");

    // iterate over all files within the path.
    set<string> sortedPaths;
    for (const directory_entry& e : directory_iterator(fullDepthSequencePath) )
        sortedPaths.insert(e.path());

    // construct and save leftDepths.
    vector<string> leftDepths(sortedPaths.begin(), sortedPaths.end());
    outputFile.writeDataset(leftDepths, sequenceH5Dir + "/leftDepths",
                            leftDepths.size(), H5D_CHUNKED);

    // loop up in the rawSequencePaths to determine root path for the leftImages
    auto leftImageRootDirIt = find_if( rawSequencePaths.begin(), rawSequencePaths.end(),
        [kittiSequenceID](const string& rawSequencePath)->bool{
            return boost::find_first(rawSequencePath, kittiSequenceID);
        });
    if (leftImageRootDirIt == rawSequencePaths.end()){
        cout << "illegeal condition: can not find " << kittiSequenceID
             << " in the list of paths of raw sequence" << endl;
        exit(-1);
    }
    string leftImageRootDir = *leftImageRootDirIt;
    cout << kittiSequenceID << " ==> " << leftImageRootDir << endl;

    // For each depth image, determine its corresponding rgb image path.
    vector<string> leftImages;
    for (string leftDepth: leftDepths){
        filesystem::path p(leftDepth);
        string leftImage = leftImageRootDir + "/image_02/data/" + p.filename().string();
        //cout << leftDepth << " -->" << leftImage << endl;
        leftImages.push_back(leftImage);
    }
    outputFile.writeDataset(leftImages, sequenceH5Dir + "/leftImages",
                            leftImages.size(), H5D_CHUNKED);

    // construct calibration data item.
    filesystem::path leftImageRootDirPath(leftImageRootDir);
    string calibrationFilePath = leftImageRootDirPath.parent_path().string() + "/"
                                 + "calib_cam_to_cam.txt";
    //cout << "calibration file path: " << calibrationFilePath << endl;
    Eigen::MatrixXf projectMatrix(3,4);
    loadLeftCameraProjectMatrix(calibrationFilePath, projectMatrix);
    outputFile.writeDataset(projectMatrix,
                            sequenceH5Dir + "/cameraCalibration/left_P_rect");
}

int main(int argc, const char*argv[])
{
    OptionsInstance instance(argc, argv);
    options::Options & ops = OptionsInstance::get();

    // load sequence paths of the depth database.
    string depthSequencePathsFilename = ops.getString("depthSequencePathList");
    if (depthSequencePathsFilename.empty()){
        cout << "missed option 'depthSequencePathsFilename' " << endl; exit(-1);
    }
    vector<string> depthSequencePaths;
    readLines(depthSequencePathsFilename, depthSequencePaths);
    cout << "read " << depthSequencePaths.size() << " sequences" << endl;

    // For unknow reason, some sequences in the depth database are missed in the raw
    // database, so we should remove them.
    deleteSomeDepthSequences(depthSequencePaths);
    cout << "after deleting some of the depth sequence, number of sequences = "
         << depthSequencePaths.size()  << endl;

    // load sequence paths of the raw database.
    string rawSequencePathsFilename = ops.getString("rawSequencePathList");
    if (rawSequencePathsFilename.empty()){
        cout << "missed option 'rawSequencePathsFilename' " << endl; exit(-1);
    }
    vector<string> rawSequencePaths;
    readLines(rawSequencePathsFilename, rawSequencePaths);
    cout << "loaded " << rawSequencePaths.size() << " paths in the raw database" << endl;

    // create the output h5 file.
    if (!ops.presents("outputFilename")){
        cout << "missed option 'outputFilename' " << endl; exit(-1);
    }
    File outputFile(ops.getString("outputFilename"), FilePermission::REPLACE);

    // construct the h5 file.
    int sequenceNumber =depthSequencePaths.size();
    outputFile.writeDataset(sequenceNumber, "/sequenceNumber");
    for (int seq=0; seq<sequenceNumber;seq++){
        string sequenceH5Dir = string("/seq_") + to_string(seq);
        processDepthSequence(depthSequencePaths[seq], sequenceH5Dir,
                             rawSequencePaths, outputFile);
    }
    cout << "have processed " << sequenceNumber << " sequences" << endl;
    return 0;
}
