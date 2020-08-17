#pragma once

#include "behaviortree_cpp_v3/behavior_tree.h"
#include "behaviortree_cpp_v3/bt_factory.h"

namespace Action {

class Track : public BT::SyncActionNode {
 public:
  Track(const std::string &name);
  BT::NodeStatus tick() override;
};

class Attack : public BT::SyncActionNode {
 public:
  Attack(const std::string &name);
  BT::NodeStatus tick() override;
};

class Dodge : public BT::SyncActionNode {
 public:
  Dodge(const std::string &name);
  BT::NodeStatus tick() override;
};

}  // namespace Action

namespace Condition {

class EnamyVisable : public BT::ConditionNode {
 public:
  EnamyVisable(const std::string &name);
  BT::NodeStatus tick() override;
};

class LowHP : public BT::ConditionNode {
 public:
  LowHP(const std::string &name);
  BT::NodeStatus tick() override;
};

class UnderAttack : public BT::ConditionNode {
 public:
  UnderAttack(const std::string &name);
  BT::NodeStatus tick() override;
};

class NoAmmo : public BT::ConditionNode {
 public:
  NoAmmo(const std::string &name);
  BT::NodeStatus tick() override;
};
}  // namespace Condition

void RegisterNode(BT::BehaviorTreeFactory &factory);
