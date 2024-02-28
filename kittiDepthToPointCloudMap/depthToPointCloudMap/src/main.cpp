#include <cmath>
#include <cstdint>
#include <iostream>
#include <Eigen/Core>
#include <h5pp/h5pp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/stereo/disparity_map_converter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include "options.h"

using namespace std;
using namespace h5pp;
using namespace options;
using namespace pcl;
using namespace cv;

template<typename T>
void printMatStatistics(const Mat& mat, string title)
{
    cout << title << endl;
    T min=numeric_limits<T>::max();
    T max=0;
    for (int row=0; row<mat.rows; row++){
        for (int col=0; col<mat.cols; col++){
            T v = mat.at<T>(row,col);
            max = std::max(v, max);
            min = std::min(v, min);
        }
    }
    cout<< "min = " << min << ", max =" << max << endl;
}

Mat loadDepthData(string leftDepth)
{
    // Load KITTI depth data
    Mat depthImg = imread(leftDepth, IMREAD_UNCHANGED);
    if (depthImg.channels()!=1 || depthImg.depth()!= CV_16U){
        cout << "Format of the input depth image file is: " << endl
             << "   channels = " << depthImg.channels() << ", depth = " << depthImg.depth() << endl
             << "but the expected one is: grayscale with a depth of 16-bits unsigned int." << endl;
        exit(-1);
    }
    cout << "Format of depth image file " << leftDepth << endl
         << "width = " << depthImg.cols << ", height = " << depthImg.rows <<", "
         << "depth = 16 bits unsigned int" << endl;

    // Display the depth data for verification.
    //ofstream debugFile("debug.txt");
    //debugFile << "depth image = " << endl
    //          << depthImg << endl;
    //cout << "content of the depth image has been writen to debug.txt" << endl;

    return depthImg;
}

void convertDepthToDisparity(const Mat& depthImg, Mat& disparityImg,
                             double baseline, double focalLength)
{
    for (int row=0; row<depthImg.rows; row++){
        for (int col=0; col<depthImg.cols; col++){
            double depth = depthImg.at<uint16_t>(row,col) / 256.0;  // in meters
            if ( fabs(depth) < 1e-6){
                // depth is zero, stand for a Not-Available value.
                disparityImg.at<float>(row,col) = -1;
            }else{
                disparityImg.at<float>(row,col) = baseline * focalLength / depth;  // in pixels.
            }
        }
    }
}

PointCloud<RGB>::Ptr loadRGBImage(string leftImage, int desiredRows, int desiredCols)
{
    // Pass an RGB image to the converter so that it can render the "I"(intensity)
    // component of the point-cloud-map. With this information, use can more eassily
    // verify whether the result is correct.
    // Since it seems that the pcl library does not offer a function to load an png image,
    // we use opencv for this.
    Mat rgbImage = imread(leftImage, IMREAD_UNCHANGED);
    if ( rgbImage.channels()!=3  || rgbImage.type() != CV_8UC3 ||
         rgbImage.rows != desiredRows ||
         rgbImage.cols != desiredCols) {
        cout << " the RGB image has a wrong number of channels, type, rows or cols" << endl;
        exit(-1);
    }
    // construct a point cloud, the following setImage function requires a parameter with
    // this type.
    PointCloud<RGB>::Ptr rgbCloud(
                new PointCloud<RGB>(rgbImage.cols, rgbImage.rows) );
    for (int row=0; row<rgbImage.rows; row++){
        for (int col=0; col<rgbImage.cols; col++){
            (*rgbCloud)(col, row).rgba = *(uint32_t*)rgbImage.ptr(row, col);
        }
    }

    return rgbCloud;
}

void processFrame(string leftDepth, string leftImage, const Eigen::MatrixXf& projMatrix,
                  string pointCloudFilename, string imagePath)
{
    Mat depthImg = loadDepthData(leftDepth);

    // Configuration parameters of the cameras
    double baseline    = 0.54;             // in meters.
    double focalLength = projMatrix(0,0);  // in pixels.
    double centerX     = projMatrix(0,2);  // in pixels.
    double centerY     = projMatrix(1,2);  // in pixels.
    //cout << "focalLength = "  << focalLength << ", "
    //     << "centerX = " << centerX << ", "
    //     << "centerY = " << centerY << endl;

    // Convert depth to disparity.
    Mat disparityImg(depthImg.rows, depthImg.cols, CV_32F);
    printMatStatistics<uint16_t>(depthImg, "statistics of depth data");
    convertDepthToDisparity(depthImg, disparityImg, baseline, focalLength);
    printMatStatistics<float>(disparityImg, "statistics of disparity data");
    //ofstream debug("debug.txt");
    //debug << disparityImg;

    DisparityMapConverter<PointXYZI> converter;
    converter.setBaseline(baseline);
    converter.setFocalLength(focalLength);
    converter.setImageCenterX(centerX);
    converter.setImageCenterY(centerY);
    converter.setDisparityThresholdMin(1.0f);

    PointCloud<RGB>::Ptr rgbCloud = loadRGBImage(leftImage, depthImg.rows, depthImg.cols);
    converter.setImage(rgbCloud);

    vector<float> disparities;
    for (int row = 0; row < disparityImg.rows; row++){
        for (int col = 0; col <disparityImg.cols; col++){
            disparities.push_back( disparityImg.at<float>(row, col) );
        }
    }
    //cout << "vector of disparity values: size = " << disparities.size() << ", "
    //     << "min = " << * std::min_element(disparities.begin(), disparities.end()) << ", "
    //     << "max = " << * std::max_element(disparities.begin(), disparities.end()) << endl;
    converter.setDisparityMap(disparities);

    PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);
    converter.compute(*cloud);

    // Quite a few of the computed points are "nan" items. They are removed so that the
    // following generated point cloud file only contains normal points.
    PointCloud<PointXYZI> conciseCloud;
    for (PointCloud<PointXYZI>::const_iterator it = cloud->begin();
         it < cloud->end(); it++){
        if (isnan(it->x) || isnan(it->y) || isnan(it->z) ) continue;
        conciseCloud.push_back( *it );
    }

    // Create parent directorie.
    filesystem::path p(pointCloudFilename);
    filesystem::create_directories(p.parent_path() );

    // write to a PLY file.
    PLYWriter plyWriter;
    plyWriter.write(pointCloudFilename, conciseCloud);
    cout << "generated ply format file " << pointCloudFilename << endl;

    // create corresponding image path file
    ofstream imagePathFile(imagePath);
    imagePathFile << leftImage << endl;
    cout << "generated image path file " << imagePath << endl;
}

void processSequence(int sequence, File& depthFile, File& cloudFile)
{
    options::Options & ops = OptionsInstance::get();

    string sequenceDir = string("seq_") + to_string(sequence);

    // print out the kittiSequenceID, just for debuging.
    string kittiSequenceID;
    depthFile.readDataset(kittiSequenceID,  sequenceDir + "/kittiSequenceID");
    cloudFile.writeDataset(kittiSequenceID, sequenceDir + "/kittiSequenceID");
    cout << "kitti ID of sequence " << sequence << ": " << kittiSequenceID << endl;

    vector<string> leftDepths;
    vector<string> leftImages;
    Eigen::MatrixXf left_P_rect;
    depthFile.readDataset(leftDepths,  sequenceDir + "/leftDepths");
    depthFile.readDataset(leftImages,  sequenceDir + "/leftImages");
    depthFile.readDataset(left_P_rect, sequenceDir + "/cameraCalibration/left_P_rect");
    assert(leftDepths.size() == leftImages.size());
    cout << "frame number: " << leftDepths.size() << endl;
    cout << "shape of project matrix: " << left_P_rect.rows() << " x " << left_P_rect.cols() << endl;
    cloudFile.writeDataset(leftImages,  sequenceDir + "/leftImages",
                           leftImages.size(), H5D_CHUNKED);

    vector<string> pointClouds;
    for (int i=0; i<leftDepths.size(); i++){
        cout << endl
             << "==== to process frame #" << i+5
             << " of sequence # " << sequence << endl;
        string pointCloud = fmt::format("{}/seq_{}/frame_{}.ply",
                                        ops.getString("pointCloudDir"),
                                        sequence, i+5);
        string imagePath =  fmt::format("{}/seq_{}/frame_{}_image_path.txt",
                                        ops.getString("pointCloudDir"),
                                        sequence, i+5);
        processFrame(leftDepths[i], leftImages[i], left_P_rect,
                     pointCloud, imagePath);
        pointClouds.push_back(pointCloud);
    }
    cloudFile.writeDataset(pointClouds,  sequenceDir + "/pointClouds",
                           pointClouds.size(), H5D_CHUNKED);
}


int main(int argc, const char* argv[])
{
    OptionsInstance instance(argc, argv);
    options::Options & ops = OptionsInstance::get();

    // Open input H5 file.
    if (!ops.presents("depthH5Filename")){
        cout << "missed option depthH5Filename" << endl; exit(-1);
    }
    File depthFile(ops.getString("depthH5Filename"), FilePermission::READONLY);

    // Create output dir.
    if (!ops.presents("pointCloudDir")){
        cout << "missed option pointCloudDir" << endl; exit(-1);
    }
    filesystem::create_directories( ops.getString("pointCloudDir"));
    // Create output H5 file.
    File cloudFile(ops.getString("pointCloudDir") + "/pointCloud.h5",
                   FilePermission::READWRITE);

    int sequenceNumber;
    depthFile.readDataset(sequenceNumber, "/sequenceNumber");
    cloudFile.writeDataset(sequenceNumber, "/sequenceNumber");
    cout << "to process " << sequenceNumber << " sequences" << endl;
    for (int i=0; i<sequenceNumber; i++){
        if (ops.presents("startFromSequence")){
            int startFromSequence = ops.getInt("startFromSequence", 0);
            if (i<startFromSequence) continue;
        }

        processSequence(i, depthFile, cloudFile);
    }
    return 0;
}

