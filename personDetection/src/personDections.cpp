#include <fmt/core.h>
#include "options.h"
#include "personDections.h"
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace Eigen;
using namespace options;

// helper function to display a Rect object.
ostream& operator<<(ostream&out, const Rect& r)
{
    out << "(x=" << r.x << ", y=" << r.y
        << ", w=" << r.width << ", h=" << r.height << ")";
    return out;
}

void splitImageIntoBlocks(const Mat& image, vector<cv::Rect>& blocks)
{
    int w = image.cols;
    int h = image.rows;
    if (w<h){
        cout << "currently this program can only process the case that image width "
                "is larger than height"  << endl;
        exit(0);
    }
    // We follow the convention that coordinates start from zero.
    int right;
    int slidingStep = h/2;
    for (right=h; right<=w; right+=slidingStep){
        Rect r;
        r.x = right - h;
        r.y = 0;
        r.width  = h;
        r.height = h;
        blocks.push_back(r);
    }
    if (right - w < slidingStep ){
        Rect r;
        r.x = w - h;
        r.y = 0;
        r.width  = h;
        r.height = h;
        blocks.push_back(r);
    }

    //cout << "the input image(" << w << "x" << h << ") "
    //     << "is split into the following blocks:" << endl;
    //for (auto r: blocks){
    //    cout << "  " << r << endl;
    //}
}

void detectPersonInBlock(Net& net, const Mat& block,
                         vector<float>& personConfidences,
                         vector<Rect>& personBoxes)
{
    options::Options & ops  = OptionsInstance::get();
    const size_t SSDWidth   = ops.getInt("SSDImageWidth",  300);
    const size_t SSDHeight  = ops.getInt("SSDImageHeight", 300);
    const float meanValue   = 127.5;
    const float scaleFactor = 0.007843f;

    Mat normalizedImage = blobFromImage(block, scaleFactor, Size(SSDWidth, SSDHeight),
                                        Scalar(meanValue, meanValue, meanValue),
                                        true/*swapRB*/, false/*crop*/);
    net.setInput(normalizedImage);

    Mat output = net.forward();
    // output the shape of the output tensor(for debug).
    //cout << "network output has a shape of (";
    //for (int i=0; i<output.dims; i++){
    //    cout << output.size[i];
    //    if (i != output.dims-1) cout << ", ";
    //}
    //cout << ")" << endl;

    // For easier access, we construct another new tensor.
    Mat cv_objects(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    //cout << cv_objects << endl;

    Eigen::MatrixXf objects;
    cv2eigen(cv_objects, objects);
    /* For the pretrained model, the class IDs are:
        0: 'background',
        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
        10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
        14: 'motorbike', 15: 'person', 16: 'pottedplant',
        17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
   */
    const int personClassID = 15;
    for (int i=0; i<objects.rows(); i++){
        // each row has a layout of (0, classID, confidence, leftTopX, leftTopY,
        // rightBottomX, rightBottomY)
        if ( (int)objects(i, 1) != personClassID) continue;
        if ( objects(i, 2) < ops.getDouble("confidenceThresholdForPerson", 0.45)) continue;

        personConfidences.push_back( objects(i, 2) );
        // Convert relative coordinates into absolute coordinates(in pixels)
        Rect r;
        r.x      = objects(i, 3) * block.cols;
        r.y      = objects(i, 4) * block.rows;
        r.width  = objects(i, 5) * block.cols - r.x;
        r.height = objects(i, 6) * block.rows - r.y;
        personBoxes.push_back(r);
    }
}

void detectPersonFromImage(Mat& image, Net& net,
                           vector<float>& personConfidences,
                           vector<Rect>& personBoxes)
{
    options::Options & ops  = OptionsInstance::get();
    vector<Rect> blockRegions;
    splitImageIntoBlocks(image, blockRegions);

    for (auto blockRegion: blockRegions){
        vector<float> confidences;
        vector<Rect> boxes;
        detectPersonInBlock(net, image(blockRegion), confidences, boxes);
        cout << "detected " << boxes.size() << " persons in region "
             << blockRegion << endl;
        //for (auto r: boxes){
        //    cout << "  " << r << endl;
        //}

        // Record the detected boxes, making its left-top-x coordinate relative to
        // the entire image.
        for (int i=0; i<boxes.size(); i++){
            personConfidences.push_back( confidences[i] );
            personBoxes.push_back( Rect( boxes[i].x + blockRegion.x,  boxes[i].y,
                                         boxes[i].width,  boxes[i].height) );
        }
    }

    // Perform an operation of NMS on the above detected result.
    cout << "before NMS, there are " << personBoxes.size() << " bboxes" << endl;
    vector<int> indices;
    NMSBoxes(personBoxes, personConfidences,
             ops.getDouble("confidenceThresholdForPerson", 0.45),
             ops.getDouble("IoUThresholdForNMS", 0.6),
             indices);
    assert(personBoxes.size() == personConfidences.size());

    for(int i=personBoxes.size()-1; i >= 0; i--){
        if ( find(indices.begin(), indices.end(), i) == indices.end()){
            personBoxes.erase(personBoxes.begin() + i);
            personConfidences.erase(personConfidences.begin() + i);
        }
    }
    cout << "after NMS, there are " << personBoxes.size() << " bboxes" << endl;
}

bool overlap(const Rect& r, const vector<Rect>& regions)
{
    for (const Rect&region: regions){
        if ( (r & region).area() > 0 ){
            return true;
        }
    }
    return false;
}

void annotate(Mat& image, const vector<float>& confidences, const vector<Rect>& boxes)
{
    vector<Rect> regions; // regions occupied by text annotations.
    for (int i=0; i<confidences.size(); i++){
        rectangle(image, boxes[i], Scalar(0, 0, 255 ),  //opencv specifiies color in BGR
                      2 );

        // We do a smart annotation: if an annotate was conflict with a previous one, shift
        // it to another position.
        string text = fmt::format("{:.2f}", confidences[i]);
        int fontFace = 0;
        double fontScale = 0.7;
        int thickness = 2;
        int baseLine;
        Size size = getTextSize(text, fontFace,  fontScale, thickness, &baseLine);
        // Determine where the text should be put.
        // Note the (x,y) members of a Rect object are the coordinate of the left-top corner of
        // the region.
        Rect initialRegion(boxes[i].x, std::max(boxes[i].y - 5 - size.height, 0),
                           size.width, size.height);
        Rect r(initialRegion);
        while (overlap(r, regions) ){
            r.x -= size.width;
            if (r.x <= 0){
                r = initialRegion;
                break;
            }
            r.y -= size.height;
            if (r.y <= 0){
                r = initialRegion;
                break;
            }
        }
        // register the new region
        regions.push_back(r);

        putText(image, text, Point(r.x, r.y+r.height),
                fontFace, fontScale,
                Scalar(0, 255, 0 ), thickness);
        if ( r != initialRegion)
            line(image, Point( boxes[i].x, boxes[i].y),  Point(r.x, r.y+r.height),
                 Scalar(0, 255, 0), thickness);
    }
}
