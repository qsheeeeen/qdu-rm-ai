#include "behavior.hpp"

#include "spdlog/spdlog.h"

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