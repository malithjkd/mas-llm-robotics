# Shared Prompt Context

This file consolidates the runtime prompt fragments and reference text that
are shared across multiple agents. Each section below is a verbatim copy of
the corresponding source file in the original WSL working tree
(`~/fyppp/test10_mas/test10_mas/`); section headers indicate the source
filename so the agent code can be updated to load this consolidated file
or the individual sections can be split back out as needed.

---


## Environment Description

_(source: `Environment_description.txt`)_

```
1. Environmental Map
Describe the map of the environment (robotics lab) including room locations and about each doors and their locations.

//content//
The environment is the robotics lab of University of Moratuwa. Environment includes 3 rooms (R1, R2, R3) and a corridor (C).
Doors are given as D1, D2, D3, D4.
Map is showing below:

    o---------o---------o----o----o
    |         |         |         |
    |   R1    D2   R2   D3   R3   |
    |         |         |         |
    o D1 o----o---------o D4 o----|
    |              C              |
    o-----------------------------o

Robot can go a full round through the map R1->R2->R3->C->R1.
The only doors which have access to the corridor are ***D1 and D4***. Example paths:
Ex: Robot need to go from R1 to R3 => It can go through R1->R2->R3 or else R1->C->R3.
Ex: Robot need to go from R2 to R3 but D3 is closed => It can go through R2->R1->C->R3 since D2 is still open.
Ex: Robot need to go from R3 to R2 but D3 and D4 closed => Error occurs, unknown.

//new section//
2. Room Descriptions
Describe about each room and for which kind of purposes each room is using and what objects each room includes.

//content//
R1: This is the study room for students. This includes the following objects in the room: sink, some tables and chairs to study, wheelchair robot, sofa.
R2: This is the tool room. You can find various kinds of tools like hammers, soldering iron, solders, circuits, power supplies and various kinds of tools. If you need to find any physical tool, you need to come here.
R3: This is the PLC room. This has 4 tables with PLC setups. And more importantly, the KUKA robot hand is in this room.
```

---

## Environment Description (Path Planner)

_(source: `Environment_description_for_PP.txt`)_

```
The environment is the robotics lab of University of Moratuwa. Environment includes 3 rooms (R1, R2, R3) and a corridor (C).
There are 4 doors (D1, D2, D3, D4). 
Map is showing below:

    o---------o---------o----o----o
    |         |         |         |
    |   R1    D2   R2   D3   R3   |
    |         |         |         |
    o D1 o----o---------o D4 o----|
    |              C              |
    o-----------------------------o

The only doors which have access to the corridor are ***D1 and D4***. 
If we go full tour in the robotics lab, path is like below.
    -- 'R1->D2->R2->D3->R3->D4->C->D1->R1'.

Ex  - If robot need to go from R3 to R2 => Path: R3->D3->R2
    - If robot need to go from R1 to R2 => Path: R1->D2->R2
    - If robot need to go from R3 to C  => Path: R3->D4->C
    - If robot need to go from R3 to D3 => Path: R3->D3

```

---

## Special Notes

_(source: `Specialnotes.txt`)_

```
In tool agent I need to know whether the agent scratchpad is keeping the memory even after we give another new call.
Since this is a service robot we need to have the ability to talk with it like asking a navigational plan kind of one.
Reset the memory at relevant places
```

---

## NHS Agent Special Notes

_(source: `Agent_NHS_special_notes.txt`)_

```
1. If the task is a navigational command like 'go to a place', you should first plan the path, next you are free to execute the robot and finish.
2. If the task has 'if door is open...'  you should first check whether the door is open or not.
```

---

## Boss: Supportive Info

_(source: `Boss_supportive_info.txt`)_

```
rules: 
The only functions which robot can support are Move to a place, identifying objects, pick and place objects and also quesion and answering.
The robot can answer questions which are related to to navigation and do tasks which are related to above functions.
for example, robot cannot perform tasks like filling water, cooking, sweeping.....But it can take you or find equipments related to the given task.

If user's request is against above rules and robots capabilities, you have to inform it to the user using speaker then Finish the task.
If the work is done successfully, you can finish the task without informing the user.
If user request is related to rules above then you can call topplanner agent. Make sure you give the correct inputs
```

---

## Top Planner: Supporting Info

_(source: `Top_planner_supporting_info.txt`)_

```
rules:  If you are asked to find an object, you have to navigate and search for the object. Make sure you put the robot current position.
        If you are asked to bring an object, you have to navigate, search , grab and comeback. Make sure you put the robot current position.
        If you are asked a question, you only have to give response to that question.

1. Example 1 {
    User request: "I need the screw driver" (task like I need it. so go and grab it for me),
    Action plan: {
        1. I'm currently in R1. Navigate to the room where I can find the screw driver.
        2. Search the object.(mention object)
        3. Grab the object.(mention object)
        4. Navigate back to the R1.
        5. Inform the user about final update (failure or success).
    }
}

2. Example 2 {
    User request: "Hii, where Can I find the kuka robot? (asking for direction)",
    Action plan: {
        1. Get the answer from the question answering agent. (may be a location or a path)
        2. Inform the answer to user.
    }
}

3. Example 3 {
    User request: "Hii robot can you take me to the kuka?" (task like take me somewhere),
    Action plan: {
        1. Inform the user (say the user to follow the robot)
        2. I'm currently in R3. Navigate to the room where I can find the screw driver.
        3. Search the object.(mention object)
        4. Inform the user about final update (failure or success (whether you found or not)).
    }
}
```

---

## Workflow Classifier: Supportive Info

_(source: `Workflowclassifier_supportive_info.txt`)_

```
rules: You have to carefully refer the task plan and the conversation history and you should decide which task to be executed next.
        -Then you have to choose the best agent to execute that task.
        -Make sure you give the correct input arguments to the agent.
        -Tasks should be completed one by one
        -If you encounter an error condition, be intelligent to skip tasks.
            - ex: If user asked to **bring an object** and if the objectsearcher cannot find that object, you can skip "grab object" step.
                But if the next step is navigate back to the starting position, you should do it before informing update to the user.
```

---

## Speaker: Supportive Info

_(source: `Speaker_supportive_info.txt`)_

```
1. If message mention about a path blockage due to all doors are closed in the path, inform user about the error message
    and also inform user that you have no capabilities on opening the door so if user can open the door, I can proceed.
2. If message is about a task completion, say that task is completed succefully.
3. If the message is about cannot find destination, say the message and also suggest to provide more details about the destination, so I can decide where to go exactly.
4. If the message is about an error, Dont mention the word 'error'
5. If the message said something about out of robot capability task, apoligise in a pleasant way and if any other thing needed, you can assist.
```

---

## Example: Final Waypoints

_(source: `Example_final_waypoints.txt`)_

```
1. if waypoints: R1->R2->R3->C, door_sequence: D2->D3->D4
    then final waypoints: R1->D2->R2->D3->R3->D4->C  
2. waypoints: C->R3->R2, door_sequence: D4->D3
    final waypoints: C->D4->R3->D3->R2 
3. waypoints: R1
    final waypoints: R1
```

---

## Example: Scenarios Plan

_(source: `Example_scenarios_plan.txt`)_

```
('R' represents rooms, 'C' represents corridor and 'D' represents doors. waypoints show the room order combined with the doors which robot
will encounter in the path.)

1. robot currently in R1 and need to go to the C and waypoints are R1->D2->R2->D3->R3->D4->C.
    navigational plan:    
    1. Go to R1.
    2. Go to D2.
    4. If D2 open, go to R2.
    5. Go to D3.
    7. If D3 open, go to R3.
    8. Go to D4.
    10. If D4 open, go to C.

2. robot currently in R3 and need to go to the R1 and waypoints are R3->D3->R2->D2->R1.
    navigational plan:    
    1. Go to R3.
    2. Go to D3.
    4. If D3 open, go to R2.
    5. Go to D2.
    7. If D2 open, go to R1.
```

---

## Error Handling Info

_(source: `Error_handling_info.txt`)_

```
1. Error 1 {
    error description: "A Door is closed.", 
    Action: {
        Next agent to call: NavigationsupervisorMain,
        agent input: a message of 'door num' is closed,
    }
}

2. Error 2 {
    error description: "The given destination cannot be identified correctly.",
    Action: {
        Next agent to call: WorkflowClassifier,
        agent input: the error message and ask the user to provide more information regarding the destination,
    }
}

3. Error 3 {
    error description: "All available paths are blocked due to the door closure.", 
    Action: {
        Next agent to call: WorkflowClassifier,
        agent input: saying all available paths are blocked. so cannot proceed further,
    }
}
```

---
