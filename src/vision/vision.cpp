#include "vision.hpp"

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void TestOpenCV(void)
{
    cout << "Test OpenCV." << endl;
    Mat img;
    img = imread("/home/qs/test.jpg", IMREAD_COLOR);
    if (img.empty())
    {
        cout << "Error opening image." << endl;
        return;
    }
    cvtColor(img, img, COLOR_BGR2GRAY);
    imwrite("/home/qs/test_result.jpg", img);
    
    cout << "Finish TestOpenCV." << endl;
}

void TestVideoWrite(void)
{
    cout << "Test TestVideoWrite." << endl;
    const string source = "/home/qs/test.mp4"; // the source file name

    VideoCapture inputVideo(source); // Open input
    if (!inputVideo.isOpened())
    {
        cout << "Could not open the input video: " << source << endl;
        return;
    }

    Size video_size = Size((int)inputVideo.get(CAP_PROP_FRAME_WIDTH), (int)inputVideo.get(CAP_PROP_FRAME_HEIGHT));

    VideoWriter outputVideo;

    outputVideo.open("/home/qs/test_result.avi", inputVideo.get(CAP_PROP_FOURCC), inputVideo.get(CAP_PROP_FPS), video_size, true);

    if (!outputVideo.isOpened())
    {
        cout << "Could not open the output video for write: " << source << endl;
        return;
    }

    Mat src, res;
    vector<Mat> spl;

    for (;;)
    {
        inputVideo >> src;
        if (src.empty())
            break;

        split(src, spl);
        cvtColor(src, spl[0], COLOR_BGR2GRAY);
        spl[1] = Mat::zeros(video_size, spl[1].type());
        spl[2] = Mat::zeros(video_size, spl[2].type());
        merge(spl, res);

        outputVideo << res;
    }

    cout << "Finish TestVideoWrite." << endl;
}

ObjectDetector::ObjectDetector(/* args */)
{
}

ObjectDetector::~ObjectDetector()
{
}
