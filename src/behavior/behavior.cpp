#include "behavior.hpp"

#include <iostream>

#include "behaviortree_cpp_v3/behavior_tree.h"
#include "behaviortree_cpp_v3/bt_factory.h"
#include "node.hpp"
#include "spdlog/spdlog.h"

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

static const char *const xml_tree_full_auto = R"(

 <root main_tree_to_execute = "MainTree" >
     <BehaviorTree ID="MainTree">
        <Fallback name="root_sequence">
            
            <Fallback name="when under attack">
                <Fallback name="when under attack">
                    
                    <NoAmmo name="no_ammo"/>

                    <Sequence name="fight_back">
                        <LowHP name="low_hp"/>
                        <Dodge name="dodge"/>
                    </Sequence>
                    
                </Fallback>

                    <UnderAttack name="under_attack"/>
                    <Dodge name="dodge"/>
            </Fallback>

            <Sequence name="fight_back">
                <LowHP name="low_hp"/>
                <Dodge name="dodge"/>
                <Track name="track"/>
                <Attack name="attack"/>
            </Sequence>

                <LowHP name="low_hp"/>
                <Track name="track"/>
                <Attack name="attack"/>
            

            <Sequence name="fight_back">
                <LowHP name="low_hp"/>
                <EnamyVisable name="enamy_visable"/>
                <Track name="track"/>
                <Attack name="attack"/>
            </Sequence>

            <Sequence name="fight">
                <EnamyVisable name="enamy_visable"/>
                <Track name="track"/>
                <Attack name="attack"/>
            </Sequence>

        </Fallback>
     </BehaviorTree>
 </root>
 
 )";

static const char *const xml_tree_auto_aim = R"(

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

void TestTree() {
  SPDLOG_INFO("Run TestTree.");

  BehaviorTreeFactory factory;

  RegisterNode(factory);

  auto tree = factory.createTreeFromText(xml_tree_test);

  tree.tickRoot();

  SPDLOG_INFO("Finish TestTree.");
}