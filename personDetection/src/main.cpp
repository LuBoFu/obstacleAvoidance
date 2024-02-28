#include <iostream>
#include <vector>
#include <h5pp/h5pp.h>
#include "options.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <Eigen/Core>
#include <boost/algorithm/string.hpp>
#include "personDections.h"

using namespace std;
using namespace options;
using namespace h5pp;
using namespace cv;
using namespace dnn;

void convertToMatrix(const vector<Rect>& boxes, MatrixXf& m)
{
    m.resize(boxes.size(), 4);
    for (int i=0; i<boxes.size(); i++){
        m(i, 0) = boxes[i].x;
        m(i, 1) = boxes[i].y;
        m(i, 2) = boxes[i].width;
        m(i, 3) = boxes[i].height;
    }
}

void processFrame(Net& net, string leftImageFile,
                  string sequenceDir, int frame,
                  File& outputFile)
{
    options::Options & ops = OptionsInstance::get();
    cout << endl
         << "to process left image: " << leftImageFile << endl;

    Mat image= imread(leftImageFile);
    vector<float> confidences;
    vector<Rect> boxes;
    detectPersonFromImage(image, net, confidences, boxes);

    // Write result to H5 file, even no person detected.
    string dataDir = sequenceDir + "/frame_" + to_string(frame);
    outputFile.writeDataset(confidences, dataDir + "/confidences", H5D_CHUNKED);
    MatrixXf boxesMatrix;
    convertToMatrix(boxes, boxesMatrix);
    outputFile.writeDataset(boxesMatrix, dataDir + "/boxes", H5D_CHUNKED);

    if (boxes.empty()){
        // no person detected, no need to annotate.
        return;
    }

    // annotate the image and save to a file.
    annotate(image, confidences, boxes);
    string annotatedImagePath = ops.getString("detectedPersonDir") + "/" +
                                sequenceDir + "/frame_" + to_string(frame) + "_annotated.png";
    cv::imwrite(annotatedImagePath, image);
}

void processSequence(Net& net, int sequence, File& inputIndex, File& outputFile)
{
    options::Options & ops = OptionsInstance::get();
    string sequenceDir = string("seq_") + to_string(sequence);

    // Read in file names of the left images.
    vector<string> leftImageFiles;
    inputIndex.readDataset(leftImageFiles,  sequenceDir + "/leftImages");
    cout << "to process sequence " << sequence
         << "(" << leftImageFiles.size() << " frames)" << endl;

    // create the sequence directory to store the annotated iamges.
    filesystem::create_directories(ops.getString("detectedPersonDir") + "/" + sequenceDir );

    // write to output file.
    outputFile.writeDataset(leftImageFiles, sequenceDir + "/leftImages",  H5D_CHUNKED);
    // process the left images.
    for (int frame=0; frame<leftImageFiles.size(); frame++){
        processFrame(net, leftImageFiles[frame],
                     sequenceDir, frame,
                     outputFile);
        if (ops.presents("numberOfLeadingFramesToProcess")){
            if ( frame+1 >= ops.getInt("numberOfLeadingFramesToProcess", 0) )
                break;
        }
    }
}

// Get the path of the first left image, determine whether it contains
// the name of the target scene. If so, return true; otherwise, return false.
bool sequenceMatchTargetScene(File& inputIndex, int sequence, string targetScene)
{
    string sequenceDir = string("seq_") + to_string(sequence);
    vector<string> leftImageFiles;
    inputIndex.readDataset(leftImageFiles,  sequenceDir + "/leftImages");
    if (leftImageFiles.empty()) return false;

    return boost::find_first(leftImageFiles.front(), targetScene);
}

void loadNetwork(string modelProtoPath, string modelParametersPath, Net& net)
{
    net = readNetFromCaffe(modelProtoPath, modelParametersPath);
    cout << "neural network model loaded" << endl;

    vector<int> outLayers = net.getUnconnectedOutLayers();
    // Currently we only process the Mobilenet-SSD model in the github
    // project "MobilNet_SSD_opencv".
    assert(outLayers.size()==1);
    string outLayerType = net.getLayer(outLayers.front())->type;
    assert(outLayerType == "DetectionOutput");
}

int main()
{
    OptionsInstance instance("config/personDetectionConfig.txt");
    options::Options & ops = OptionsInstance::get();

    Net net;
    loadNetwork(ops.getString("modelProto"), ops.getString("modelParameters"), net);

    if (ops.presents("processSampleImageOnly")){
        Mat image= imread(ops.getString("sampleImagePath"));
        vector<float> confidences;
        vector<Rect> boxes;
        detectPersonFromImage(image, net, confidences, boxes);

        // annotate the image and save to a file.
        annotate(image, confidences, boxes);
        string renderedImageName = "personsInSampleImage.png";
        cv::imwrite(renderedImageName, image);
        return 0;
    }

    File inputIndex(ops.getString("kittiRawFilteredDir") + "/kittiRawFilteredIndex.h5",
                    FilePermission::READONLY);
    filesystem::create_directories(ops.getString("detectedPersonDir"));
    File outputFile(ops.getString("detectedPersonDir") + "/detectedPerson.h5",
                    FilePermission::REPLACE);

    int sequenceNumber;
    inputIndex.readDataset(sequenceNumber, "/sequenceNumber");
    cout << "to process " << sequenceNumber << " sequences" << endl;
    outputFile.writeDataset(sequenceNumber, "/sequenceNumber");

    for (int i=0; i<sequenceNumber; i++){
        if (ops.presents("targetScene")){
            if ( !sequenceMatchTargetScene(inputIndex, i, ops.getString("targetScene")))
                continue;
        }

        processSequence(net, i, inputIndex, outputFile);

        if (ops.presents("numberOfSequencesToProcess")){
            static int sequenceCounter = 0;
            if (++sequenceCounter >= ops.getInt("numberOfSequencesToProcess", 0))
                break;
        }
    }
    return 0;
}
