#include "camera.hpp"

#include <cstring>
#include <exception>
#include <string>
#include <thread>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#include "spdlog/spdlog.h"

void Camera::PrintDeviceInfo(MV_CC_DEVICE_INFO *mv_dev_info) {
  if (nullptr == mv_dev_info) {
    SPDLOG_ERROR("[Camera] The Pointer of mv_dev_info is nullptr!\n");
    return;
  }
  if (mv_dev_info->nTLayerType == MV_USB_DEVICE) {
    SPDLOG_INFO(
        "[Camera] UserDefinedName: {} \nSerial Number: {}\nDevice Number: {}",
        mv_dev_info->SpecialInfo.stUsb3VInfo.chUserDefinedName,
        mv_dev_info->SpecialInfo.stUsb3VInfo.chSerialNumber,
        mv_dev_info->SpecialInfo.stUsb3VInfo.nDeviceNumber);
  } else {
    SPDLOG_WARN("[Camera] Not support.");
  }
}

void Camera::Prepare() {
  int err = MV_OK;
  std::string err_msg;
  SPDLOG_DEBUG("[Camera] Prepare.");

  std::memset(&mv_dev_list_, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
  err = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &mv_dev_list_);
  if (err != MV_OK) {
    err_msg = "[Camera] EnumDevices fail! err: " + std::to_string(err);
    SPDLOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  if (mv_dev_list_.nDeviceNum > 0) {
    for (unsigned int i = 0; i < mv_dev_list_.nDeviceNum; ++i) {
      SPDLOG_INFO("[Camera] Device {d} slected.", i);
      MV_CC_DEVICE_INFO *dev_info = mv_dev_list_.pDeviceInfo[i];
      if (dev_info == nullptr) {
        SPDLOG_ERROR("[Camera] Error Reading dev_info");
        throw std::runtime_error("[Camera] Error Reading dev_info");
      } else
        PrintDeviceInfo(dev_info);
    }
  } else {
    SPDLOG_ERROR("[Camera] Find No Devices!");
    throw std::runtime_error("[Camera] Find No Devices!");
  }
}

Camera::Camera(unsigned int out_h, unsigned int out_w)
    : out_h_(out_h), out_w_(out_w) {
  SPDLOG_DEBUG("[Camera] Constructing.");
  Prepare();
  SPDLOG_DEBUG("[Camera] Constructed.");
}

Camera::Camera(unsigned int index, unsigned int out_h, unsigned int out_w)
    : out_h_(out_h), out_w_(out_w) {
  SPDLOG_DEBUG("[Camera] Constructing.");
  Prepare();
  Open(index);
  SPDLOG_DEBUG("[Camera] Constructed.");
}

Camera::~Camera() {
  SPDLOG_DEBUG("[Camera] Destructing.");
  Close();
  SPDLOG_DEBUG("[Camera] Destructed.");
}

int Camera::Open(unsigned int index) {
  int err = MV_OK;
  std::string err_msg;

  SPDLOG_DEBUG("[Camera] Open index:{}.", index);

  if (index >= mv_dev_list_.nDeviceNum) {
    SPDLOG_ERROR("[Camera] Intput index:{} >= nDeviceNum:{} !", index,
                 mv_dev_list_.nDeviceNum);
    return MV_E_UNKNOW;
  }

  err = MV_CC_CreateHandle(&camera_handle_, mv_dev_list_.pDeviceInfo[index]);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] CreateHandle fail! err: {}", err);
    return err;
  }

  err = MV_CC_OpenDevice(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] OpenDevice fail! err: {}", err);
    return err;
  }

  err = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", 0);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] SetTrigger fail! err: {}", err);
    return err;
  }

  err = MV_CC_StartGrabbing(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] StartGrabbing fail! err: {}", err);
    return err;
  }
  return MV_OK;
}

cv::Mat Camera::GetFrame() {
  int err = MV_OK;
  MV_FRAME_OUT frame_out = {};

  cv::Mat image;

  if (!MV_CC_IsDeviceConnected(camera_handle_)) {
    SPDLOG_ERROR("[Camera] Camera disconnected.");
    goto finally;
  }
  err = MV_CC_GetImageBuffer(camera_handle_, &frame_out, 10);
  if (err == MV_OK) {
    size_t rgb_size =
        frame_out.stFrameInfo.nWidth * frame_out.stFrameInfo.nHeight * 4 + 2048;

    std::vector<unsigned char> raw_rgb(rgb_size);

    MV_CC_PIXEL_CONVERT_PARAM cvt_param{
        .nWidth = frame_out.stFrameInfo.nWidth,
        .nHeight = frame_out.stFrameInfo.nHeight,

        .enSrcPixelType = frame_out.stFrameInfo.enPixelType,
        .pSrcData = frame_out.pBufAddr,
        .nSrcDataLen = frame_out.stFrameInfo.nFrameLen,

        .enDstPixelType = PixelType_Gvsp_RGB8_Packed,
        .pDstBuffer = raw_rgb.data(),
        .nDstBufferSize = static_cast<unsigned int>(rgb_size),
    };

    SPDLOG_DEBUG(
        "[Camera] Get One Frame: Width{d}, Height{d}, "
        "nFrameNum{d}\n",
        frame_out.stFrameInfo.nWidth, frame_out.stFrameInfo.nHeight,
        frame_out.stFrameInfo.nFrameNum);

    err = MV_CC_ConvertPixelType(camera_handle_, &cvt_param);
    if (MV_OK != err) {
      SPDLOG_ERROR("[Camera] ConvertPixelType fail! err:{x}\n", err);
      goto finally;
    }

    cv::Mat raw(
        cv::Size(frame_out.stFrameInfo.nWidth, frame_out.stFrameInfo.nHeight),
        CV_8UC3, raw_rgb.data());

    const int offset_h = (raw.rows - raw.cols) / 2;
    const cv::Rect roi(offset_h, 0, raw.cols, raw.cols);
    cv::resize(raw(roi), image, cv::Size(out_h_, out_w_));

  } else {
    SPDLOG_ERROR("[Camera] GetImageBuffer fail! err:{x}\n", err);
  }

finally:
  if (NULL != frame_out.pBufAddr) {
    err = MV_CC_FreeImageBuffer(camera_handle_, &frame_out);
    if (err != MV_OK) {
      SPDLOG_ERROR("[Camera] FreeImageBuffer fail! err:{x}\n", err);
    }
  }
  return image;
}

int Camera::Close() {
  int err = MV_OK;
  std::string err_msg;

  SPDLOG_DEBUG("[Camera] Close.");

  err = MV_CC_StopGrabbing(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] StopGrabbing fail! err:{x}", err);
    return err;
  }

  err = MV_CC_CloseDevice(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] ClosDevice fail! err:{x}", err);
    return err;
  }

  err = MV_CC_DestroyHandle(camera_handle_);
  if (err != MV_OK) {
    SPDLOG_ERROR("[Camera] DestroyHandle fail! err:{x}", err);
    return err;
  }

  SPDLOG_DEBUG("[Camera] Closed.");
  return MV_OK;
}
