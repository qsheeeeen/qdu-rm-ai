#include "vision.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"

using namespace cv;
using namespace std;

void TestOpenCV(void)
{
    std::cout << "Test OpenCV." << std::endl;
    Mat img;
    img = imread("/home/qs/test.jpg", IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Error opening image." << std::endl;
        return;
    }
    cvtColor(img, img, COLOR_BGR2GRAY);
    imwrite("/home/qs/result.jpg", img);
}

void TestVideoWrite(void)
{
    const string source = "/home/qs/test.avi"; // the source file name

    VideoCapture inputVideo(source); // Open input
    if (!inputVideo.isOpened())
    {
        cout << "Could not open the input video: " << source << endl;
        return;
    }

    string::size_type pAt = source.find_last_of('.');           // Find extension point
    const string NAME = "/home/qs/test_result.avi";             // Form the new name with container
    int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC)); // Get Codec Type- Int form

    // Transform from int to char via Bitwise operators
    char EXT[] = {(char)(ex & 0XFF), (char)((ex & 0XFF00) >> 8), (char)((ex & 0XFF0000) >> 16), (char)((ex & 0XFF000000) >> 24), 0};

    Size S = Size((int)inputVideo.get(CAP_PROP_FRAME_WIDTH), // Acquire input size
                  (int)inputVideo.get(CAP_PROP_FRAME_HEIGHT));

    VideoWriter outputVideo; // Open the output

    outputVideo.open(NAME, ex, inputVideo.get(CAP_PROP_FPS), S, true);

    if (!outputVideo.isOpened())
    {
        cout << "Could not open the output video for write: " << source << endl;
        return;
    }

    cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
         << " of nr#: " << inputVideo.get(CAP_PROP_FRAME_COUNT) << endl;
    cout << "Input codec type: " << EXT << endl;

    int channel = 2; // Select the channel to save
    Mat src, res;
    vector<Mat> spl;

    for (;;) //Show the image captured in the window and repeat
    {
        inputVideo >> src; // read
        if (src.empty())
            break; // check if at end

        split(src, spl); // process - extract only the correct channel
        for (int i = 0; i < 3; ++i)
            if (i != channel)
                spl[i] = Mat::zeros(S, spl[0].type());
        merge(spl, res);

        //outputVideo.write(res); //save or
        outputVideo << res;
    }

    cout << "Finished writing" << endl;
}

void TestOpenCVGraph(void)
{
    std::cout << "Test OpenCV Graph." << std::endl;

    VideoCapture video_in("/home/qs/test.avi");

    if (!video_in.isOpened())
    {
        cout << "Could not open the input video: " << endl;
        return;
    }

    int h = video_in.get(CAP_PROP_FRAME_HEIGHT);
    int w = video_in.get(CAP_PROP_FRAME_WIDTH);

    VideoWriter video_out;
    video_out.open("/home/qs/result.avi", video_in.get(CAP_PROP_FOURCC), video_in.get(CAP_PROP_FPS), Size(w, h), true);

    if (!video_out.isOpened())
    {
        cout << "Could not open the output video for write: " << endl;
        return;
    }

    Mat input_frame;
    Mat output_frame;

    while (video_in.read(input_frame))
    {
        cvtColor(input_frame, output_frame, COLOR_BGR2GRAY);
        video_out.write(output_frame);
    }
}

ObjectDetector::ObjectDetector(/* args */)
{
}

ObjectDetector::~ObjectDetector()
{
}
