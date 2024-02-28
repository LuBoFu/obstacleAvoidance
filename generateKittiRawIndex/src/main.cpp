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
    //cout << "from " << filename << ", read matrix: " << endl
    //     << m << endl;
}

int processSequence(string sequencePath, string sequenceH5Dir,
                    File& indexFile)
{
    using namespace filesystem;
    cout << endl << "processing " << sequencePath << endl;

    // get sequence kitti ID. sequencePath looks like:
    // /sw/cv/dbase/kittiRaw/Residential/2011_09_26/2011_09_26_drive_0019_sync
    filesystem::path p(sequencePath);
    string kittiSequenceID = p.filename().string();
    indexFile.writeDataset( kittiSequenceID, sequenceH5Dir + "/kittiSequenceID");

    // iterate over all files within the directory of left images, to get their paths.
    set<string> sortedPaths;
    for (const directory_entry& e : directory_iterator(sequencePath + "/image_02/data") )
        sortedPaths.insert(e.path());
    vector<string> leftImagePaths(sortedPaths.begin(), sortedPaths.end());

    // construct and verify paths for the right images.
    vector<string> rightImagePaths(leftImagePaths);
    for(string&p: rightImagePaths){
        boost::replace_first(p, "/image_02/", "/image_03/");
        if ( !filesystem::exists(p)){
            cout << "illegal condition: path " << p << " does not exits" << endl;
            exit(-1);
        }
    }

    // write out.
    indexFile.writeDataset(leftImagePaths, sequenceH5Dir + "/leftImages",
                            leftImagePaths.size(), H5D_CHUNKED);
    indexFile.writeDataset(rightImagePaths, sequenceH5Dir + "/rightImages",
                            rightImagePaths.size(), H5D_CHUNKED);

    // construct calibration data item.
    string calibrationFilePath = filesystem::path(sequencePath).parent_path().string() +
                                     "/calib_cam_to_cam.txt";
    //cout << "calibration file path: " << calibrationFilePath << endl;
    Eigen::MatrixXf projectMatrix(3,4);
    loadLeftCameraProjectMatrix(calibrationFilePath, projectMatrix);
    indexFile.writeDataset(projectMatrix,
                           sequenceH5Dir + "/cameraCalibration/left_P_rect");

    return leftImagePaths.size();
}

int main(int argc, const char*argv[])
{
    OptionsInstance instance(argc, argv);
    options::Options & ops = OptionsInstance::get();

    // load list of drive relative pathes.
    vector<string> drivePathList;
    readLines(ops.getString("drivePathList"), drivePathList);
    cout << "loaded " << drivePathList.size() << " drive paths" << endl;

    // append the root directory to make the path absolute
    for (string& path: drivePathList){
        path = ops.getString("kittiRawRoot") + "/" + path;
    }

    // create the h5 index file.
    File indexFile(ops.getString("indexFile"), FilePermission::REPLACE);

    int sequenceNumber = drivePathList.size();
    indexFile.writeDataset(sequenceNumber, "/sequenceNumber");

    int framePairNumber = 0;
    for (int seq=0; seq<sequenceNumber;seq++){
        string sequenceH5Dir = string("/seq_") + to_string(seq);
        framePairNumber += processSequence(drivePathList[seq], sequenceH5Dir,
                                           indexFile);
    }
    cout << "Totally processed " << sequenceNumber << " sequences, "
         << framePairNumber << " frame pairs " << endl;

    return 0;
}
