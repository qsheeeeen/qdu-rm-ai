#include "camera.hpp"

#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <thread>

void Camera::WorkThread() {
  int err = MV_OK;
  MV_FRAME_OUT frame_out;
  std::memset(&frame_out, 0, sizeof(MV_FRAME_OUT));
  while (continue_capture_) {
    err = MV_CC_GetImageBuffer(camera_handle_, &frame_out, 1000);
    if (err == MV_OK) {
      printf("Get One Frame: Width[%d], Height[%d], nFrameNum[%d]\n",
             frame_out.stFrameInfo.nWidth, frame_out.stFrameInfo.nHeight,
             frame_out.stFrameInfo.nFrameNum);
    } else {
      printf("No data[0x%x]\n", err);
    }
    if (NULL != frame_out.pBufAddr) {
      err = MV_CC_FreeImageBuffer(camera_handle_, &frame_out);
      if (err != MV_OK) {
        printf("Free Image Buffer fail! err [0x%x]\n", err);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void Camera::PrintDeviceInfo() {
  if (NULL == mv_dev_info_) {
    std::cout << "The Pointer of mv_dev_info_ is NULL!\n" << std::endl;
    return;
  }
  if (mv_dev_info_->nTLayerType == MV_USB_DEVICE) {
    std::cout << "UserDefinedName: "
              << mv_dev_info_->SpecialInfo.stUsb3VInfo.chUserDefinedName << "\n"
              << "Serial Number: "
              << mv_dev_info_->SpecialInfo.stUsb3VInfo.chSerialNumber << "\n"
              << "Device Number: "
              << mv_dev_info_->SpecialInfo.stUsb3VInfo.nDeviceNumber << "\n"
              << std::endl;
  } else {
    std::cout << "Not support." << std::endl;
  }
}

Camera::Camera(unsigned int index) {
  int err = MV_OK;
  std::stringstream err_string;

  std::cout << "Create Camera." << std::endl;

  std::memset(&mv_dev_list_, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
  err = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &mv_dev_list_);
  if (err != MV_OK) {
    err_string << "Enum Devices fail! err 0x" << std::hex << err << std::endl;
    throw std::runtime_error(err_string.str());
  }

  if (mv_dev_list_.nDeviceNum > 0) {
    for (unsigned int i = 0; i < mv_dev_list_.nDeviceNum; i++) {
      std::cout << "[device %d]: " << i << std::endl;
      mv_dev_info_ = mv_dev_list_.pDeviceInfo[i];
      if (mv_dev_info_ == nullptr)
        throw std::runtime_error("Error Reading mv_dev_info_");
      else
        PrintDeviceInfo();
    }
  } else {
    throw std::runtime_error("Find No Devices!");
  }

  if (index >= mv_dev_list_.nDeviceNum) {
    throw std::runtime_error("Intput error!");
  }
  err = MV_CC_CreateHandle(&camera_handle_, mv_dev_list_.pDeviceInfo[index]);
  if (err != MV_OK) {
    err_string << "Create Handle fail! err:0x" << std::hex << err << std::endl;
    throw std::runtime_error(err_string.str());
  }
  err = MV_CC_OpenDevice(camera_handle_);
  if (err != MV_OK) {
    err_string << "Open Device fail! err:0x" << std::hex << err << std::endl;
    throw std::runtime_error(err_string.str());
  }
  err = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", 0);
  if (err != MV_OK) {
    err_string << "Set Trigger Mode fail! err:0x" << std::hex << err
               << std::endl;
    throw std::runtime_error(err_string.str());
  }

  memset(&init_val_, 0, sizeof(MVCC_INTVALUE));
  err = MV_CC_GetIntValue(camera_handle_, "PayloadSize", &init_val_);
  if (err != MV_OK) {
    err_string << "Get PayloadSize fail! err:0x" << std::hex << err
               << std::endl;
    throw std::runtime_error(err_string.str());
  }

  err = MV_CC_StartGrabbing(camera_handle_);
  if (err != MV_OK) {
    err_string << "Start Grabbing fail! err:0x" << std::hex << err << std::endl;
    throw std::runtime_error(err_string.str());
  }

  continue_capture_ = true;
  capture_thread_ = std::thread(&Camera::WorkThread, this);
}

Camera::~Camera() {
  int err = MV_OK;
  std::stringstream err_string;

  continue_capture_ = false;
  capture_thread_.join();

  err = MV_CC_StopGrabbing(camera_handle_);
  if (err != MV_OK) {
    std::cout << "Stop Grabbing fail! err:0x" << std::hex << err << std::endl;
  }
  err = MV_CC_CloseDevice(camera_handle_);
  if (err != MV_OK) {
    std::cout << "ClosDevice fail! err:0x" << std::hex << err << std::endl;
  }
  err = MV_CC_DestroyHandle(camera_handle_);
  if (err != MV_OK) {
    std::cout << "Destroy Handle fail! err:0x" << std::hex << err << std::endl;
  }

  std::cout << "Camera Destried." << std::endl;
}

bool Camera::GetFrame(void* output) { return false; }
