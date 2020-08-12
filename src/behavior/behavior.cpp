#include "behavior.hpp"

#include <iostream>

#include "behaviortree_cpp_v3/behavior_tree.h"
#include "behaviortree_cpp_v3/bt_factory.h"

#include "node.hpp"

static const char *xml_text = R"(

 <root main_tree_to_execute = "MainTree" >
     <BehaviorTree ID="MainTree">
        <Sequence name="root_sequence">
            <ApproachObject name="approach1"/>
            <ApproachObject name="approach2"/>
            <ApproachObject name="approach3"/>
            <ApproachObject name="approach4"/>
        </Sequence>
     </BehaviorTree>
 </root>
 
 )";

using namespace BT;

void RunTestTree()
{
    std::cout << "Run test tree." << std::endl;

    BehaviorTreeFactory factory;
    factory.registerNodeType<ApproachObject>("ApproachObject");

    auto tree = factory.createTreeFromText(xml_text);

    tree.tickRoot();
}