#include "camera.hpp"

#include <exception>
#include <iostream>
#include <thread>

#include "opencv2/opencv.hpp"

using namespace cv;

static void WorkThread(Camera& camera) {
  int err = MV_OK;
  MV_FRAME_OUT frame_out;
  std::memset(&frame_out, 0, sizeof(MV_FRAME_OUT));
  while (1) {
    err = MV_CC_GetImageBuffer(camera.GetCameraHandle(), &frame_out, 1000);
    if (err == MV_OK) {
      printf("Get One Frame: Width[%d], Height[%d], nFrameNum[%d]\n",
             frame_out.stFrameInfo.nWidth, frame_out.stFrameInfo.nHeight,
             frame_out.stFrameInfo.nFrameNum);
    } else {
      printf("No data[0x%x]\n", err);
    }
    if (NULL != frame_out.pBufAddr) {
      err = MV_CC_FreeImageBuffer(camera.GetCameraHandle(), &frame_out);
      if (err != MV_OK) {
        printf("Free Image Buffer fail! err [0x%x]\n", err);
      }
    }
  }
}

void Camera::LoadTargetHist() {
  frame_in = imread("./image/hist_target.jpg");
  if (frame_in.empty())
    return;
  else
    calcHist(&frame_in, 1, hist_ch, Mat(), hist_target, 3, hist_size,
             hist_range);
}

void Camera::MatchTargetHist() {
  calcHist(&frame_in, 1, hist_ch, Mat(), hist_in, 3, hist_size, hist_range);
  double min, max;
  minMaxIdx(hist_in, &min, &max);
}

void Camera::CreateLUT() {
  lut.create(1, 256, CV_8U);

  for (int i = 0; i < 256; i++)
    lut.at<float>(i) = saturate_cast<uint8_t>(pow(i / 255.0, gamma) * 255.0);
}

void Camera::AppyLUT() { LUT(frame_in, lut, frame_in); }

Camera::Camera(unsigned int index) {
  int err = MV_OK;
  std::stringstream err_string;

  std::cout << "Create Camera." << std::endl;

  video_cap.open(index);
  // this->LoadTargetHist();
  // video_cap.open(index);

  std::memset(&mv_dev_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
  err = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &mv_dev_list);
  if (err != MV_OK) {
    err_string << "Enum Devices fail! err [0x%x]" << err << std::endl;
    throw std::runtime_error(err_string.str());
  }

  if (mv_dev_list.nDeviceNum > 0) {
    for (unsigned int i = 0; i < mv_dev_list.nDeviceNum; i++) {
      std::cout << "[device %d]: " << i << std::endl;
      mv_dev_info = mv_dev_list.pDeviceInfo[i];
      if (mv_dev_info == nullptr)
        throw std::runtime_error("Error Reading mv_dev_info");
      else
        PrintDeviceInfo();
    }
  } else {
    throw std::runtime_error("Find No Devices!");
  }

  if (index >= mv_dev_list.nDeviceNum) {
    throw std::runtime_error("Intput error!");
  }
  err = MV_CC_CreateHandle(&camera_handle, mv_dev_list.pDeviceInfo[index]);
  if (err != MV_OK) {
    err_string << "Create Handle fail! err:0x" << std::hex << err << std::endl;
    throw std::runtime_error(err_string.str());
  }
  err = MV_CC_OpenDevice(camera_handle);
  if (err != MV_OK) {
    err_string << "Open Device fail! err:0x" << std::hex << err << std::endl;
    throw std::runtime_error(err_string.str());
  }
  err = MV_CC_SetEnumValue(camera_handle, "TriggerMode", 0);
  if (err != MV_OK) {
    err_string << "Set Trigger Mode fail! err:0x" << std::hex << err
               << std::endl;
    throw std::runtime_error(err_string.str());
  }

  memset(&init_val, 0, sizeof(MVCC_INTVALUE));
  err = MV_CC_GetIntValue(camera_handle, "PayloadSize", &init_val);
  if (err != MV_OK) {
    err_string << "Get PayloadSize fail! err:0x" << std::hex << err
               << std::endl;
    throw std::runtime_error(err_string.str());
  }

  payload_size = init_val.nCurValue;
  err = MV_CC_StartGrabbing(camera_handle);
  if (err != MV_OK) {
    err_string << "Start Grabbing fail! err:0x" << std::hex << err << std::endl;
    throw std::runtime_error(err_string.str());
  }

  std::thread capture_thread(WorkThread, std::ref(*this));
}

Camera::~Camera() {
  int err = MV_OK;
  std::stringstream err_string;

  video_cap.release();

  if (camera_handle != nullptr) {
    err = MV_CC_StopGrabbing(camera_handle);
    if (err != MV_OK) {
      std::cout << "Stop Grabbing fail! err:0x" << std::hex << err << std::endl;
    }
    err = MV_CC_CloseDevice(camera_handle);
    if (err != MV_OK) {
      std::cout << "ClosDevice fail! err:0x" << std::hex << err << std::endl;
    }
    err = MV_CC_DestroyHandle(camera_handle);
    if (err != MV_OK) {
      std::cout << "Destroy Handle fail! err:0x" << std::hex << err
                << std::endl;
    }
  }

  std::cout << "Camera Destried." << std::endl;
}

void* Camera::GetCameraHandle() { return this->camera_handle; }

bool Camera::Capture() {
  video_cap >> frame_in;
  return frame_in.empty();
}

void Camera::Preprocess() {
  this->AppyLUT();
  this->MatchTargetHist();
}

void Camera::GetFrame(Mat& output) { frame_in.copyTo(output); }

void Camera::CalcTargetHist() {}

void Camera::PrintDeviceInfo() {
  if (NULL == mv_dev_info) {
    std::cout << "The Pointer of mv_dev_info is NULL!" << std::endl;
    return;
  }
  if (mv_dev_info->nTLayerType == MV_GIGE_DEVICE) {
    int nIp1 =
        ((mv_dev_info->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
    int nIp2 =
        ((mv_dev_info->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
    int nIp3 =
        ((mv_dev_info->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
    int nIp4 = (mv_dev_info->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);
    std::cout << "CurrentIp: " << nIp1 << nIp2 << nIp3 << nIp4 << "";
    std::cout << "UserDefinedName: "
              << mv_dev_info->SpecialInfo.stGigEInfo.chUserDefinedName
              << std::endl;
  } else if (mv_dev_info->nTLayerType == MV_USB_DEVICE) {
    std::cout << "UserDefinedName: "
              << mv_dev_info->SpecialInfo.stUsb3VInfo.chUserDefinedName << "\n";
    std::cout << "Serial Number: "
              << mv_dev_info->SpecialInfo.stUsb3VInfo.chSerialNumber << "\n";
    std::cout << "Device Number: "
              << mv_dev_info->SpecialInfo.stUsb3VInfo.nDeviceNumber
              << std::endl;
  } else {
    std::cout << "Not support." << std::endl;
  }
}