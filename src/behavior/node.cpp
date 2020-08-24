#include "node.hpp"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#include "spdlog/spdlog.h"

namespace Action {

Track::Track(const std::string& name) : BT::SyncActionNode(name, {}) {
  SPDLOG_DEBUG("[Action::Track] Create: {}.", name);
}

BT::NodeStatus Track::tick() {
  SPDLOG_DEBUG("[Action::Track] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
}

Attack::Attack(const std::string& name) : BT::SyncActionNode(name, {}) {
  SPDLOG_DEBUG("[Action::Attack] Create: {}.", name);
}

BT::NodeStatus Attack::tick() {
  SPDLOG_DEBUG("[Action::Attack] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
}

Dodge::Dodge(const std::string& name) : BT::SyncActionNode(name, {}) {
  SPDLOG_DEBUG("[Action::Dodge] Create: {}.", name);
}

BT::NodeStatus Dodge::tick() {
  SPDLOG_DEBUG("[Action::Dodge] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
}

}  // namespace Action

namespace Condition {

EnamyVisable::EnamyVisable(const std::string& name)
    : BT::ConditionNode(name, {}) {
  SPDLOG_DEBUG("[Condition::EnamyVisable] Create: {}.", name);
}

BT::NodeStatus EnamyVisable::tick() {
  SPDLOG_DEBUG("[Condition::EnamyVisable] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
}

LowHP::LowHP(const std::string& name) : BT::ConditionNode(name, {}) {
  SPDLOG_DEBUG("[Condition::LowHP] Create: {}.", name);
}

BT::NodeStatus LowHP::tick() {
  SPDLOG_DEBUG("[Condition::LowHP] {} ticking.", this->name());
  return BT::NodeStatus::FAILURE;
}

UnderAttack::UnderAttack(const std::string& name)
    : BT::ConditionNode(name, {}) {
  SPDLOG_DEBUG("[Condition::LowHP] Create: {}.", name);
}

BT::NodeStatus UnderAttack::tick() {
  SPDLOG_DEBUG("[Condition::LowHP] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
}

NoAmmo::NoAmmo(const std::string& name) : BT::ConditionNode(name, {}) {
  SPDLOG_DEBUG("[Condition::NoAmmo] Create: {}.", name);
}

BT::NodeStatus NoAmmo::tick() {
  SPDLOG_DEBUG("[Condition::LowHP] {} ticking.", this->name());
  return BT::NodeStatus::SUCCESS;
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