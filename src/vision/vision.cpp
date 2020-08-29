#include "vision.hpp"

#include "opencv2/opencv.hpp"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include "spdlog/spdlog.h"

using namespace cv;
using namespace std;

void TestOpenCV(void) {
  SPDLOG_INFO("Test OpenCV.");

  Mat img;
  img = imread("/home/qs/test.jpg", IMREAD_COLOR);
  if (img.empty()) {
    SPDLOG_ERROR("Error opening image.");
    return;
  }
  cvtColor(img, img, COLOR_BGR2GRAY);
  imwrite("/home/qs/test_result.jpg", img);

  SPDLOG_INFO("Finish TestOpenCV.");
}

void TestVideoWrite(void) {
  SPDLOG_INFO("Test TestVideoWrite.");
  const string source = "/home/qs/test.mp4";

  VideoCapture inputVideo(source);
  if (!inputVideo.isOpened()) {
    SPDLOG_ERROR("Could not open the input video: {}.", source);
    return;
  }

  Size video_size = Size((int)inputVideo.get(CAP_PROP_FRAME_WIDTH),
                         (int)inputVideo.get(CAP_PROP_FRAME_HEIGHT));

  VideoWriter outputVideo;

  outputVideo.open("/home/qs/test_result.avi", inputVideo.get(CAP_PROP_FOURCC),
                   inputVideo.get(CAP_PROP_FPS), video_size, true);

  if (!outputVideo.isOpened()) {
    SPDLOG_ERROR("Could not open the output video for write: {}",source );
    return;
  }

  Mat src, res;
  vector<Mat> spl;

  for (;;) {
    inputVideo >> src;
    if (src.empty()) break;

    split(src, spl);
    cvtColor(src, spl[0], COLOR_BGR2GRAY);
    spl[1] = Mat::zeros(video_size, spl[1].type());
    spl[2] = Mat::zeros(video_size, spl[2].type());
    merge(spl, res);

    outputVideo << res;
  }

  SPDLOG_INFO("Finish TestVideoWrite.");
}
