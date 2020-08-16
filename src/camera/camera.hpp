#pragma once

#include "opencv2/opencv.hpp"

class Camera
{
private:
    cv::VideoCapture cam;
    cv::Mat frame_in;
    cv::Mat hist_in;
    cv::Mat hist_target;

    const int hist_ch[3] = {0, 1, 2};
    const int hist_size[3] = {256, 256, 256};
    float hist_ch_range[2] = {0., 255.};
    const float *hist_range[3] = {hist_ch_range, hist_ch_range, hist_ch_range};

    float gamma = 1.;
    cv::Mat lut;

    void CreateLUT();
    void AppyLUT();

    void LoadTargetHist();

    void GetFrame(cv::Mat &output);

    void MatchTargetHist();

public:
    Camera(int index);
    ~Camera();
    bool Capture();
    void Preprocess();
    void CalcTargetHist();
};
