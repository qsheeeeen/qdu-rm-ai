#include "node.hpp"

namespace Action
{
    BT::NodeStatus Track::tick()
    {
        std::cout << this->name() << ": Tracking." << std::endl;
        return BT::NodeStatus::SUCCESS;
    }

    BT::NodeStatus Attack::tick()
    {
        std::cout << this->name() << ": Attacking." << std::endl;
        return BT::NodeStatus::SUCCESS;
    }

} // namespace Action

namespace Condition
{
    BT::NodeStatus EnamyVisable::tick()
    {
        std::cout << this->name() << ": Check EnamyVisable." << std::endl;
        return BT::NodeStatus::SUCCESS;
    }

    BT::NodeStatus LowHP::tick()
    {
        std::cout << this->name() << ": Check LowHP." << std::endl;
        return BT::NodeStatus::FAILURE;
    }

    BT::NodeStatus UnderAttack::tick()
    {
        std::cout << this->name() << ": Check UnderAttack." << std::endl;
        return BT::NodeStatus::SUCCESS;
    }
} // namespace Condition
