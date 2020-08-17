#include "node.hpp"

namespace Action {

Track::Track(const std::string& name) : BT::SyncActionNode(name, {}) {
  std::cout << "Creat Action::Track node: " << this->name() << std::endl;
}

BT::NodeStatus Track::tick() {
  std::cout << this->name() << ": Tracking." << std::endl;
  return BT::NodeStatus::SUCCESS;
}

Attack::Attack(const std::string& name) : BT::SyncActionNode(name, {}) {
  std::cout << "Creat Action::Attack node: " << this->name() << std::endl;
}

BT::NodeStatus Attack::tick() {
  std::cout << this->name() << ": Attacking." << std::endl;
  return BT::NodeStatus::SUCCESS;
}

Dodge::Dodge(const std::string& name) : BT::SyncActionNode(name, {}) {
  std::cout << "Creat Action::Dodge node: " << this->name() << std::endl;
}

BT::NodeStatus Dodge::tick() {
  std::cout << this->name() << ": Dodgeing." << std::endl;
  return BT::NodeStatus::SUCCESS;
}

}  // namespace Action

namespace Condition {

EnamyVisable::EnamyVisable(const std::string& name)
    : BT::ConditionNode(name, {}) {
  std::cout << "Creat Condition::EnamyVisable node: " << this->name()
            << std::endl;
}

BT::NodeStatus EnamyVisable::tick() {
  std::cout << this->name() << ": Check EnamyVisable." << std::endl;
  return BT::NodeStatus::SUCCESS;
}

LowHP::LowHP(const std::string& name) : BT::ConditionNode(name, {}) {
  std::cout << "Creat Condition::LowHP node: " << this->name() << std::endl;
}

BT::NodeStatus LowHP::tick() {
  std::cout << this->name() << ": Check LowHP." << std::endl;
  return BT::NodeStatus::FAILURE;
}

UnderAttack::UnderAttack(const std::string& name)
    : BT::ConditionNode(name, {}) {
  std::cout << "Creat Condition::UnderAttack node: " << this->name()
            << std::endl;
}

BT::NodeStatus UnderAttack::tick() {
  std::cout << this->name() << ": Check UnderAttack." << std::endl;
  return BT::NodeStatus::SUCCESS;
}

NoAmmo::NoAmmo(const std::string& name) : BT::ConditionNode(name, {}) {
  std::cout << "Creat Condition::NoAmmo node: " << this->name() << std::endl;
}

BT::NodeStatus NoAmmo::tick() {
  std::cout << this->name() << ": Check NoAmmo." << std::endl;
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