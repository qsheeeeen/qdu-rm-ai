#include "robot.hpp"

#include <fstream>

void Robot::WorkThread() {
    
}

Robot::Robot(std::string dev_path) {
  dev = std::ofstream(dev_path);
  continue_parse_ = true;
  parse_thread_ = std::thread(&Robot::WorkThread, this);
}

Robot::~Robot() {
    dev.close();
}

bool Robot::Connect() {}
bool Robot::Disconnect() {}

bool Robot::Parse() {}
bool Robot::Send() {}
