#pragma once

#include "behaviortree_cpp_v3/behavior_tree.h"

// Example of custom SyncActionNode (synchronous action)
// without ports.
class ApproachObject : public BT::SyncActionNode
{
public:
    ApproachObject(const std::string &name) : BT::SyncActionNode(name, {}) {
        std::cout << "Init ApproachObject action node: ";
        std::cout << this->name();
        std::cout << std::endl;
    }

    BT::NodeStatus tick() override;
};