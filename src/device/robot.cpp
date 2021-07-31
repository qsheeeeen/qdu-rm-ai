#include "robot.hpp"

#include "opencv2/core/quaternion.hpp"
#include "spdlog/spdlog.h"

namespace {

const double kLIMIT = 0.04;

}  // namespace

void Robot::ThreadRecv() {
  SPDLOG_DEBUG("[ThreadRecv] Started.");

  Protocol_ID_t id;
  Protocol_UpPackageReferee_t ref;
  Protocol_UpPackageMCU_t robot;

  while (thread_continue) {
    serial_.Recv(&id, sizeof(id));

    if (AI_ID_REF == id) {
      serial_.Recv(&ref, sizeof(ref));

      if (crc16::CRC16_Verify((uint8_t *)&ref, sizeof(ref))) {
        mutex_ref_.lock();
        std::memcpy(&ref_, &(ref.data), sizeof(ref_));
        mutex_ref_.unlock();
      }
    } else if (AI_ID_MCU == id) {
      serial_.Recv(&robot, sizeof(robot));

      if (crc16::CRC16_Verify((uint8_t *)&robot, sizeof(robot))) {
        mutex_mcu_.lock();
        std::memcpy(&mcu_, &(robot.data), sizeof(mcu_));
        mutex_mcu_.unlock();
      }
    }
  }
  SPDLOG_DEBUG("[ThreadRecv] Stoped.");
}

void Robot::ThreadTrans() {
  SPDLOG_DEBUG("[ThreadTrans] Started.");

  Protocol_DownPackage_t command;

  while (thread_continue) {
    mutex_commandq_.lock();
    if (!commandq_.empty()) {
      command.data = commandq_.front();
      command.crc16 = crc16::CRC16_Calc((uint8_t *)&command.data,
                                        sizeof(command.data), UINT16_MAX);
      serial_.Trans((char *)&command, sizeof(command));
      commandq_.pop();
    }
    mutex_commandq_.unlock();
  }
  SPDLOG_DEBUG("[ThreadTrans] Stoped.");
}

Robot::Robot() { SPDLOG_TRACE("Constructed."); }

Robot::Robot(const std::string &dev_path) {
  Init(dev_path);

  SPDLOG_TRACE("Constructed.");
}

Robot::~Robot() {
  serial_.Close();

  thread_continue = false;
  thread_recv_.join();
  thread_trans_.join();
  SPDLOG_TRACE("Destructed.");
}

void Robot::Init(const std::string &dev_path) {
  serial_.Open(dev_path);
  serial_.Config();
  if (!serial_.IsOpen()) {
    SPDLOG_ERROR("Can't open device.");
  }

  thread_continue = true;
  thread_recv_ = std::thread(&Robot::ThreadRecv, this);
  thread_trans_ = std::thread(&Robot::ThreadTrans, this);
}

game::Team Robot::GetTeam() {
  if (ref_.team == AI_TEAM_RED)
    return game::Team::kBLUE;
  else if (ref_.team == AI_TEAM_BLUE)
    return game::Team::kRED;
  return game::Team::kUNKNOWN;
}

double Robot::GetTime() { return 90 - ref_.time; }

cv::Mat Robot::GetRotMat() {
  cv::Quatf q(mcu_.quat.q0, mcu_.quat.q1, mcu_.quat.q2, mcu_.quat.q3);
  return cv::Mat(q.toRotMat3x3(), true);
}

void Robot::Aim(component::Euler aiming_eulr, bool auto_fire) {
  data_.gimbal.pit = aiming_eulr.pitch;
  data_.gimbal.rol = aiming_eulr.roll;
  data_.gimbal.yaw = aiming_eulr.yaw;

  // TODO

  double w = mcu_.quat.q0, x = mcu_.quat.q1, y = mcu_.quat.q2, z = mcu_.quat.q3;
  component::Euler euler;

  const float sinr_cosp = 2.0f * (w * x + y * z);
  const float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
  euler.pitch = atan2f(sinr_cosp, cosr_cosp);

  const float sinp = 2.0f * (w * y - z * x);

  if (fabsf(sinp) >= 1.0f)
    euler.roll = copysignf(CV_PI / 2.0f, sinp);
  else
    euler.roll = asinf(sinp);

  const float siny_cosp = 2.0f * (w * z + x * y);
  const float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
  euler.yaw = atan2f(siny_cosp, cosy_cosp);

  if (!auto_fire)
    data_.notice |= 0x07;
  else {
    if (fabs(euler.pitch - aiming_eulr.pitch) >= kLIMIT)
      ;
    else if (fabs(euler.roll - aiming_eulr.roll) >= kLIMIT)
      ;
    else if (fabs(euler.yaw - aiming_eulr.pitch) >= kLIMIT)
      ;
    else
      data_.notice |= AI_NOTICE_FIRE;
  }

  commandq_.push(data_);
}

void Robot::Move() {}
