#include "camera.hpp"

#include <iostream>

#include "opencv2/opencv.hpp"

using namespace cv;

void Camera::LoadTargetHist()
{
    frame_in = imread("./image/hist_target.jpg");
    calcHist(&frame_in, 1, hist_ch, Mat(), hist_target, 3, hist_size, hist_range);
}

void Camera::MatchTargetHist()
{
    calcHist(&frame_in, 1, hist_ch, Mat(), hist_in, 3, hist_size, hist_range);
    double min, max;
    minMaxIdx(hist_in, &min, &max);
}

void Camera::CreateLUT()
{
    lut.create(1, 256, CV_8U);

    for (int i = 0; i < 256; i++)
        lut.at<float>(i) = saturate_cast<uint8_t>(pow(i / 255.0, gamma) * 255.0);
}

void Camera::AppyLUT()
{
    LUT(frame_in, lut, frame_in);
}

Camera::Camera(int index)
{
    std::cout << "Create Camera." << std::endl;

    this->CreateLUT();
    this->LoadTargetHist();

    cam.open(index, CAP_V4L2);
}

Camera::~Camera()
{
    cam.release();
    std::cout << "Camera Destried." << std::endl;
}

bool Camera::Capture()
{
    cam >> frame_in;
    return frame_in.empty();
}

void Camera::Preprocess()
{
    this->AppyLUT();
    this->MatchTargetHist();
}

void Camera::GetFrame(cv::Mat &output)
{
    frame_in.copyTo(output);
}

void Camera::CalcTargetHist()
{
}