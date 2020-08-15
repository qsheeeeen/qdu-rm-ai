#pragma once

#include "opencv2/opencv.hpp"

class Camera
{
private:
    cv::VideoCapture cap;

public:
    Camera(/* args */);
    ~Camera();
    void Capture();
    void Preprocess();
};
