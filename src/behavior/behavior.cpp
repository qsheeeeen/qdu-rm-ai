#include "behavior.hpp"

#include <iostream>

#include "behaviortree_cpp_v3/behavior_tree.h"
#include "behaviortree_cpp_v3/bt_factory.h"

#include "node.hpp"

static const char *xml_text = R"(

 <root main_tree_to_execute = "MainTree" >
     <BehaviorTree ID="MainTree">
        <Fallback name="root_sequence">
            <LowHP name="low_hp"/>
            <Sequence name="fight">
                <EnamyVisable name="enamy_visable"/>
                <Track name="track"/>
                <Attack name="attack"/>
            </Sequence>
        </Fallback>
     </BehaviorTree>
 </root>
 
 )";

using namespace BT;

void TestTree()
{
    std::cout << "Run TestTree." << std::endl;

    BehaviorTreeFactory factory;
    factory.registerNodeType<Action::Attack>("Attack");
    factory.registerNodeType<Action::Track>("Track");
    factory.registerNodeType<Condition::EnamyVisable>("EnamyVisable");
    factory.registerNodeType<Condition::LowHP>("LowHP");
    factory.registerNodeType<Condition::UnderAttack>("UnderAttack");

    auto tree = factory.createTreeFromText(xml_text);

    tree.tickRoot();
    
    std::cout << "Finish TestTree." << std::endl;
}