#include "camera.hpp"

#include <iostream>

#include "opencv2/opencv.hpp"

using namespace cv;

Camera::Camera(/* args */)
{
    std::cout << "Create Camera." << std::endl;
}

Camera::~Camera()
{
    std::cout << "Destroy Camera." << std::endl;
}