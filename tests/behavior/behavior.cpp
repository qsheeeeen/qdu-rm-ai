#include <iostream>

#include "behaviortree_cpp_v3/behavior_tree.h"
#include "behaviortree_cpp_v3/bt_factory.h"
#include "gtest/gtest.h"
#include "node.hpp"

TEST(TestBehavior, ExampleTest) { EXPECT_EQ(1, 1); }

static const char *const xml_tree_test = R"(

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

TEST(TestBehavior, TestSimpleTree) {
  BT::BehaviorTreeFactory factory;

  RegisterNode(factory);

  auto tree = factory.createTreeFromText(xml_tree_test);

  tree.tickRoot();
}