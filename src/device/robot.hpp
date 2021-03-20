#pragma once

#include <mutex>
#include <queue>
#include <thread>

#include "common.hpp"
#include "crc16.hpp"
#include "opencv2/opencv.hpp"
#include "protocol.h"
#include "serial.hpp"

class Robot {
 private:
  Serial serial_;
  bool thread_continue = false;
  std::thread thread_recv_, thread_trans_;

  std::queue<Protocol_DownData_t> commandq_;
  Protocol_UpDataReferee_t ref_;
  Protocol_UpDataMCU_t mcu_;
  Protocol_DownData_t data_;

  std::mutex mutex_commandq_, mutex_ref_, mutex_mcu_;

  void ThreadRecv();
  void ThreadTrans();

 public:
  Robot(const std::string &dev_path);
  ~Robot();

  game::Team GetTeam();
  cv::Mat GetRotMat();
  void Aim(common::Euler aiming_eulr, bool auto_fire);
  void Move();
};
