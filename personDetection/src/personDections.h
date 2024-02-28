#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <Eigen/Core>

using std::string;
using std::vector;
using cv::dnn::Net;
using cv::Mat;
using cv::Rect;
using Eigen::MatrixXf;

void detectPersonFromImage(Mat& image, Net& net,
                           vector<float>& personConfidences,
                           vector<Rect>& personBoxes);

void annotate(Mat& image, const vector<float>& confidences, const vector<Rect>& boxes);
