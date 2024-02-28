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

Mat loadDisparityData(string disparityPath)
{
    // Load disparity data
    Mat disparityImg = imread(disparityPath, IMREAD_UNCHANGED);
    if (disparityImg.channels()!=1 || disparityImg.depth()!= CV_16U){
        cout << "Format of the input disparity image file is: " << endl
             << "   channels = " << disparityImg.channels()
             << ", depth = " << disparityImg.depth() << endl
             << "but the expected one is: grayscale with a depth of 16-bits unsigned int." << endl;
        exit(-1);
    }
    cout << "Format of disparity image file " << disparityPath << endl
         << "width = " << disparityImg.cols << ", height = " << disparityImg.rows <<", "
         << "depth = 16 bits unsigned int" << endl;

    // Display the data for verification.
    //ofstream debugFile("debug.txt");
    //debugFile << "disparity image = " << endl
    //          << disparityImg << endl;
    //cout << "content of the disparty image has been writen to debug.txt" << endl;

    return disparityImg;
}



PointCloud<RGB>::Ptr loadRGBImage(string leftImage, int desiredRows, int desiredCols)
{
    // Pass an RGB image to the converter so that it can render the "I"(intensity)
    // component of the point-cloud-map. With this information, it is easier for the user
    // to check the result.
    // Since it seems that the pcl library does not offer a function to load an png image,
    // we use opencv for this.
    Mat rgbImage = imread(leftImage, IMREAD_UNCHANGED);
    if ( rgbImage.channels()!=3  || rgbImage.type() != CV_8UC3 ||
         rgbImage.rows != desiredRows ||
         rgbImage.cols != desiredCols) {
        cout << " the RGB image has a wrong number of channels, type, rows or cols" << endl;
        exit(-1);
    }
    // construct a point cloud
    PointCloud<RGB>::Ptr rgbCloud(
                new PointCloud<RGB>(rgbImage.cols, rgbImage.rows) );
    for (int row=0; row<rgbImage.rows; row++){
        for (int col=0; col<rgbImage.cols; col++){
            (*rgbCloud)(col, row).rgba = *(uint32_t*)rgbImage.ptr(row, col);
        }
    }

    return rgbCloud;
}

void processFrame(string disparityPath, string leftImagePath, const Eigen::MatrixXf& projMatrix,
                  string pointCloudPath, string imagePathFilename)
{
    Mat disparityImg = loadDisparityData(disparityPath);
    printMatStatistics<uint16_t>(disparityImg, "statistics of disparity data");


    // Configuration parameters of the cameras
    double baseline    = 0.54;             // in meters.
    double focalLength = projMatrix(0,0);  // in pixels.
    double centerX     = projMatrix(0,2);  // in pixels.
    double centerY     = projMatrix(1,2);  // in pixels.
    //cout << "focalLength = "  << focalLength << ", "
    //     << "centerX = " << centerX << ", "
    //     << "centerY = " << centerY << endl;

    DisparityMapConverter<PointXYZI> converter;
    converter.setBaseline(baseline);
    converter.setFocalLength(focalLength);
    converter.setImageCenterX(centerX);
    converter.setImageCenterY(centerY);
    converter.setDisparityThresholdMin(1.0f);

    PointCloud<RGB>::Ptr rgbCloud = loadRGBImage(leftImagePath, disparityImg.rows, disparityImg.cols);
    converter.setImage(rgbCloud);

    vector<float> disparities;
    for (int row = 0; row < disparityImg.rows; row++){
        for (int col = 0; col <disparityImg.cols; col++){
            float disparityValue = disparityImg.at<uint16_t>(row, col) / 256.0;
            disparities.push_back(disparityValue);
        }
    }
    cout << "vector of disparity values: size = " << disparities.size() << ", "
         << "min = " << * std::min_element(disparities.begin(), disparities.end()) << ", "
         << "max = " << * std::max_element(disparities.begin(), disparities.end()) << endl;
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
    filesystem::path p(pointCloudPath);
    filesystem::create_directories(p.parent_path() );

    // write to a PLY file.
    PLYWriter plyWriter;
    plyWriter.write(pointCloudPath, conciseCloud);
    cout << "generated ply format file " << pointCloudPath << endl;

    // create corresponding image path file
    ofstream imagePathFile(imagePathFilename);
    imagePathFile << leftImagePath << endl;
    cout << "generated image path file " << imagePathFilename << endl;
}

void processSequence(int sequence, File& indexFile)
{
    options::Options & ops = OptionsInstance::get();
    vector<string> disparityImages;
    vector<string> leftImages;
    Eigen::MatrixXf left_P_rect;
    string sequenceDir = string("seq_") + to_string(sequence);
    indexFile.readDataset(disparityImages,  sequenceDir + "/disparityImages");
    indexFile.readDataset(leftImages,  sequenceDir + "/leftImages");
    indexFile.readDataset(left_P_rect, sequenceDir + "/cameraCalibration/left_P_rect");
    assert(disparityImages.size() == leftImages.size());
    cout << "frame number: " << disparityImages.size() << endl;
    cout << "shape of project matrix: " << left_P_rect.rows() << " x " << left_P_rect.cols() << endl;

    vector<string> pointClouds;
    for (int i=0; i<disparityImages.size(); i++){
        cout << endl
             << "==== to process frame #" << i << "/" << disparityImages.size()
             << " of sequence # " << sequence << endl;
        string pointCloudPath = fmt::format("{}/seq_{}/frame_{}.ply",
                                        ops.getString("pointCloudDir"),
                                        sequence, i);
        string imagePathFile =  fmt::format("{}/seq_{}/frame_{}_image_path.txt",
                                        ops.getString("pointCloudDir"),
                                        sequence, i);
        processFrame(disparityImages[i], leftImages[i], left_P_rect,
                     pointCloudPath, imagePathFile);
        pointClouds.push_back(pointCloudPath);
    }
    indexFile.writeDataset(pointClouds,  sequenceDir + "/pointClouds",
                           pointClouds.size(), H5D_CHUNKED);
}


int main(int argc, const char* argv[])
{
    OptionsInstance instance(argc, argv);
    options::Options & ops = OptionsInstance::get();

    File indexFile(ops.getString("kittiRawIndex"), FilePermission::READWRITE);
    filesystem::create_directories( ops.getString("pointCloudDir"));    

    int sequenceNumber;
    indexFile.readDataset(sequenceNumber, "/sequenceNumber");
    cout << "to process " << sequenceNumber << " sequences" << endl;
    for (int i=0; i<sequenceNumber; i++){
        processSequence(i, indexFile);
    }
    return 0;
}

