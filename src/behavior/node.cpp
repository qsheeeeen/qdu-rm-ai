#include "node.hpp"

namespace Action {

BT::NodeStatus Track::tick() {
  std::cout << this->name() << ": Tracking." << std::endl;
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus Attack::tick() {
  std::cout << this->name() << ": Attacking." << std::endl;
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus Dodge::tick() {
  std::cout << this->name() << ": Dodgeing." << std::endl;
  return BT::NodeStatus::SUCCESS;
}

}  // namespace Action

namespace Condition {

BT::NodeStatus EnamyVisable::tick() {
  std::cout << this->name() << ": Check EnamyVisable." << std::endl;
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus LowHP::tick() {
  std::cout << this->name() << ": Check LowHP." << std::endl;
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus UnderAttack::tick() {
  std::cout << this->name() << ": Check UnderAttack." << std::endl;
  return BT::NodeStatus::SUCCESS;
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