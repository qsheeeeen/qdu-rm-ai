#include "node.hpp"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#include "robot.hpp"
#include "spdlog/spdlog.h"

namespace Action {

Track::Track(const std::string& name, const BT::NodeConfiguration& node_cfg)
    : BT::SyncActionNode(name, node_cfg) {
  SPDLOG_DEBUG("[Action::Track] Constructe: {}.", name);
}

BT::NodeStatus Track::tick() {
  SPDLOG_DEBUG("[Action::Track] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS; 
}

BT::PortsList Track::providedPorts() {
  return {BT::OutputPort<std::string>("text")};
}

Attack::Attack(const std::string& name, const BT::NodeConfiguration& node_cfg)
    : BT::SyncActionNode(name, node_cfg) {
  SPDLOG_DEBUG("[Action::Attack] Constructe: {}.", name);
}

BT::NodeStatus Attack::tick() {
  SPDLOG_DEBUG("[Action::Attack] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
}

BT::PortsList Attack::providedPorts() {
  return {BT::OutputPort<std::string>("text")};
}

Dodge::Dodge(const std::string& name, const BT::NodeConfiguration& node_cfg)
    : BT::SyncActionNode(name, node_cfg) {
  SPDLOG_DEBUG("[Action::Dodge] Constructe: {}.", name);
}

BT::NodeStatus Dodge::tick() {
  SPDLOG_DEBUG("[Action::Dodge] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
}

BT::PortsList Dodge::providedPorts() {
  return {BT::OutputPort<std::string>("text")};
}

}  // namespace Action

namespace Condition {

EnamyVisable::EnamyVisable(const std::string& name,
                           const BT::NodeConfiguration& node_cfg)
    : BT::ConditionNode(name, {}) {
  SPDLOG_DEBUG("[Condition::EnamyVisable] Constructe: {}.", name);
}

BT::NodeStatus EnamyVisable::tick() {
  SPDLOG_DEBUG("[Condition::EnamyVisable] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
}

BT::PortsList EnamyVisable::providedPorts() {
  return {BT::OutputPort<std::string>("text")};
}

LowHP::LowHP(const std::string& name, const BT::NodeConfiguration& node_cfg)
    : BT::ConditionNode(name, {}) {
  SPDLOG_DEBUG("[Condition::LowHP] Constructe: {}.", name);
}

BT::NodeStatus LowHP::tick() {
  SPDLOG_DEBUG("[Condition::LowHP] {} ticking.", this->name());
  return BT::NodeStatus::FAILURE;
}

BT::PortsList LowHP::providedPorts() {
  return {BT::OutputPort<std::string>("text")};
}

UnderAttack::UnderAttack(const std::string& name,
                         const BT::NodeConfiguration& node_cfg)
    : BT::ConditionNode(name, {}) {
  SPDLOG_DEBUG("[Condition::LowHP] Constructe: {}.", name);
}

BT::NodeStatus UnderAttack::tick() {
  SPDLOG_DEBUG("[Condition::LowHP] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
}

BT::PortsList UnderAttack::providedPorts() {
  return {BT::OutputPort<std::string>("text")};
}

NoAmmo::NoAmmo(const std::string& name, const BT::NodeConfiguration& node_cfg)
    : BT::ConditionNode(name, {}) {
  SPDLOG_DEBUG("[Condition::NoAmmo] Constructe: {}.", name);
}

BT::NodeStatus NoAmmo::tick() {
  SPDLOG_DEBUG("[Condition::LowHP] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
}

BT::PortsList NoAmmo::providedPorts() {
  return {BT::OutputPort<std::string>("text")};
}

}  // namespace Condition

void RegisterNode(BT::BehaviorTreeFactory& factory) {
  factory.registerNodeType<Action::Attack>("Attack");
  factory.registerNodeType<Action::Track>("Track");
  factory.registerNodeType<Action::Dodge>("Dodge");

  factory.registerNodeType<Condition::EnamyVisable>("EnamyVisable");
  factory.registerNodeType<Condition::LowHP>("LowHP");
  factory.registerNodeType<Condition::UnderAttack>("UnderAttack");
  factory.registerNodeType<Condition::NoAmmo>("NoAmmo");
}