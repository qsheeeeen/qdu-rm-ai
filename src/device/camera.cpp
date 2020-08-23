#include "camera.hpp"

#include <cstring>
#include <exception>
#include <sstream>
#include <thread>

#include "spdlog/spdlog.h"

void Camera::WorkThread() {
  spdlog::debug("[Camera][WorkThread] Running.");

  int err = MV_OK;
  MV_FRAME_OUT frame_out;
  std::memset(&frame_out, 0, sizeof(MV_FRAME_OUT));
  while (continue_capture_) {
    err = MV_CC_GetImageBuffer(camera_handle_, &frame_out, 1000);
    if (err == MV_OK) {
      spdlog::info("Get One Frame: Width{d}, Height{d}, nFrameNum{d}\n",
                   frame_out.stFrameInfo.nWidth, frame_out.stFrameInfo.nHeight,
                   frame_out.stFrameInfo.nFrameNum);
    } else {
      spdlog::error("GetImageBuffer fail! err:{x}\n", err);
    }
    if (NULL != frame_out.pBufAddr) {
      err = MV_CC_FreeImageBuffer(camera_handle_, &frame_out);
      if (err != MV_OK) {
        spdlog::error("FreeImageBuffer fail! err:{x}\n", err);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  spdlog::debug("[Camera][WorkThread] Running.");
}

void Camera::PrintDeviceInfo() {
  if (NULL == mv_dev_info_) {
    spdlog::warn("[Camera] The Pointer of mv_dev_info_ is NULL!\n");
    return;
  }
  if (mv_dev_info_->nTLayerType == MV_USB_DEVICE) {
    spdlog::info(
        "[Camera] UserDefinedName: {} \nSerial Number: {}\nDevice Number: {}",
        mv_dev_info_->SpecialInfo.stUsb3VInfo.chUserDefinedName,
        mv_dev_info_->SpecialInfo.stUsb3VInfo.chSerialNumber,
        mv_dev_info_->SpecialInfo.stUsb3VInfo.nDeviceNumber);
  } else {
    spdlog::warn("[Camera] Not support.");
  }
}

Camera::Camera(unsigned int index) {
  int err = MV_OK;
  std::stringstream err_string;

  spdlog::debug("[Camera] Creating.");

  std::memset(&mv_dev_list_, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
  err = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &mv_dev_list_);
  if (err != MV_OK) {
    err_string << "[Camera] Enum Devices fail! err 0x" << std::hex << err
               << std::endl;
    spdlog::error(err_string.str());
    throw std::runtime_error(err_string.str());
  }

  if (mv_dev_list_.nDeviceNum > 0) {
    for (unsigned int i = 0; i < mv_dev_list_.nDeviceNum; i++) {
      spdlog::info("[device {d}]: ", i);
      mv_dev_info_ = mv_dev_list_.pDeviceInfo[i];
      if (mv_dev_info_ == nullptr) {
        spdlog::error(err_string.str());
        throw std::runtime_error("[Camera] Error Reading mv_dev_info_");
      } else
        PrintDeviceInfo();
    }
  } else {
    spdlog::error(err_string.str());
    throw std::runtime_error("[Camera] Find No Devices!");
  }

  if (index >= mv_dev_list_.nDeviceNum) {
    spdlog::error(err_string.str());
    throw std::runtime_error("[Camera] Intput error!");
  }
  err = MV_CC_CreateHandle(&camera_handle_, mv_dev_list_.pDeviceInfo[index]);
  if (err != MV_OK) {
    err_string << "[Camera] Create Handle fail! err:0x" << std::hex << err
               << std::endl;
    spdlog::error(err_string.str());
    throw std::runtime_error(err_string.str());
  }
  err = MV_CC_OpenDevice(camera_handle_);
  if (err != MV_OK) {
    err_string << "[Camera] Open Device fail! err:0x" << std::hex << err
               << std::endl;
    spdlog::error(err_string.str());
    throw std::runtime_error(err_string.str());
  }
  err = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", 0);
  if (err != MV_OK) {
    err_string << "[Camera] Set Trigger Mode fail! err:0x" << std::hex << err
               << std::endl;
    spdlog::error(err_string.str());
    throw std::runtime_error(err_string.str());
  }

  memset(&init_val_, 0, sizeof(MVCC_INTVALUE));
  err = MV_CC_GetIntValue(camera_handle_, "PayloadSize", &init_val_);
  if (err != MV_OK) {
    err_string << "[Camera] Get PayloadSize fail! err:0x" << std::hex << err
               << std::endl;
    spdlog::error(err_string.str());
    throw std::runtime_error(err_string.str());
  }

  err = MV_CC_StartGrabbing(camera_handle_);
  if (err != MV_OK) {
    err_string << "[Camera] Start Grabbing fail! err:0x" << std::hex << err
               << std::endl;
    spdlog::error(err_string.str());
    throw std::runtime_error(err_string.str());
  }

  continue_capture_ = true;
  capture_thread_ = std::thread(&Camera::WorkThread, this);

  spdlog::debug("[Camera] Created.");
}

Camera::~Camera() {
  int err = MV_OK;
  std::stringstream err_string;

  spdlog::debug("[Camera] Destroying.");

  continue_capture_ = false;
  capture_thread_.join();

  err = MV_CC_StopGrabbing(camera_handle_);
  if (err != MV_OK) spdlog::error("[Camera] StopGrabbing fail! err:{x}", err);

  err = MV_CC_CloseDevice(camera_handle_);
  if (err != MV_OK) spdlog::error("[Camera] ClosDevice fail! err:{x}", err);

  err = MV_CC_DestroyHandle(camera_handle_);
  if (err != MV_OK) spdlog::error("[Camera] DestroyHandle fail! err:{x}", err);

  spdlog::debug("[Camera] Destried.");
}

bool Camera::GetFrame(void* output) { return false; }
