"""All in one test 1"""

#--------------MULTI-AGENT-SYSTEM-----------------
import os
import time
import pandas as pd
from dotenv import load_dotenv

"""Import robot files"""
import robot3

"""Import the agents"""
from mas.agents.transcriptioner import *
from mas.agents.boss import *
from mas.agents.top_planner import *
from mas.agents.speaker import *
from mas.agents.workflow_classifier import *
from mas.agents.object_searcher import *
from mas.agents.question_answerer import *
from mas.agents.navigational_supervisor_main import *
from mas.agents.navigational_supervisor_sec import *
from mas.agents.final_destination_identifier import *
from mas.agents.waypoint_generator import *
from mas.agents.final_navigational_planner import *
from mas.agents.navigation_handler_main import *
from mas.agents.navigation_handler_sec import *
from mas.agents.path_planner import *
from mas.agents.door_status_checker import *
from mas.agents.error_handler import *

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langchain_tracking_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

"""State"""
from typing import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        response_workflow_classifier: Response from the workflow classifier
        response_navigation_supervisor_main: Response from the navigational supervisor main
    """
    # general parameters
    Current_robot_position_G: str
    Recent_agent_G: str
    Raw_user_request_G: str
    Transcriptioner_status: str

    # parameters of the boss
    Thought_boss: str
    Next_agent_boss: str
    AgentInput_boss: str
    # parameters of the speaker
    Speaker_caller: str
    Output_speaker: str
    # parameters of the top planner
    Thought_toppl: str
    taskplan_toppl: str
    # parameters of the workflow classifier
    Thought_wc: str
    Next_agent_wc: str
    AgentInput_wc: str
    # parameters of the object searcher
    Object_tobe_searched_os: str
    Object_search_status: str
    Object_location_os: str
    # parameters of the object grabber
    Object_tobe_grabbed_og: str
    Object_location_og: str
    Object_grabber_status: str
    # parameters of the question answerer
    Explanation_qa: str
    FinalAnswer_qa: str
    # parameters of the navigational supervisor main
    response_navigation_supervisor_main: str
    # parameters of the navigational supervisor sec - explanation, next agent, agent inputs
    Thought_navsupsec: str
    Action_navsupsec: str
    AgentInput_navsupsec: str
    # parameters of destination decisioner
    Explanation_destination: str
    # parameters of current position identifier
    Current_robot_position_CPI: str
    # parameters of waypoint generator
    Room_waypoints: str
    Door_sequence: str
    Final_waypoints: str
    # parameters of final navigational planner
    Plan_FNP: str
    # parameters of navigation handler main - Next task in the schedule, Router
    Next_task_NHM: str
    Action_type_NHM: str
    # parameters of navigation handler secondary
    Thought_navhansec: str
    Action_navhansec: str
    AgentInput_navhansec: str
    # parameters of door checker
    Door_status_explanation: str
    # parameters of path planner
    Path_planning_status: str
    Target_orientation: int
    Target_position: str
    # parameters of robot executor
    Robot_executor_status: str

    # parameters of error handler
    Thought_error: str
    Action_error: str
    AgentInput_error: str
    # parameters of the finisher
    check_finisher: str


"""Activate the robot"""
Robot = robot3.Robot(current_position=(195, 195), show_rt_display=True, show_rt_camera_frame=True)
print("Robot is initialised\n")

"""Activate the agents"""
Transcriptioner = TranscriptionAgent()
Boss = BossAgent()
TopPlanner = TopPlannerAgent()
Speaker = SpeakerAgent()
WorkflowClassifier = WorkflowClassifierAgent()
ObjectSearcher = ObjectFindingAgent()
QuestionAnswerer = QuestionAnsweringAgent()
NavigationSupervisorMain = NavigationSupervisorMainAgent()
NavigationSupervisorSec = NavigationSupervisorSecAgent()
DestinationIdentifier = DestinationDecisionAgent()
WaypointGenerator = WaypointGenerationAgent()
FinalNavigationPlanner = FinalNavigationalPlannerAgent()
NavigationHandlerMain = NavigationalHandlerMainAgent()
NavigationHandlerSec = NavigationalHandlerSecAgent()
PathPlanner = PathPlannerAgent()
DoorStatusChecker = DoorStatusCheckerAgent()
ErrorHandler = ErrorHandlerAgent()

"""for tracking the execution time of the agents"""
# List to store timing data
timing_data = []

# Function to reset timing data
def reset_timing_data():
    global timing_data
    timing_data = []

# Function wrapper to record execution time
def time_function(agent_name, start_time, end_time):
    start = start_time
    end = end_time
    total = round(end - start, 6)

    # Add to timing data
    timing_data.append({
        'Agent Name': agent_name,
        'Starting Time (s)': round(start, 6),
        'Ending Time (s)': round(end, 6),
        'Total Time Taken (s)': total
    })

"""Agents"""
# Speech agent
def Speech_agent(state: GraphState):
    print("\n---SPEECH AGENT---\n")

    start_time = time.time()

    transcription_status = state["Transcriptioner_status"]

    user_request = input("Enter your request: ")

    # if transcription_status == "off":
    #     user_request = str(input("Enter your request: "))
    # elif transcription_status == "on":
    #     switch = input("Push Enter to listen")
    #     user_request_obj = Transcriptioner.get_transcription_from_mic()
    #     user_request = user_request_obj.transcribed_text

    print(f"User request: {user_request}")

    end_time = time.time()
    time_function("SpeechAgent", start_time, end_time)

    return {"Raw_user_request_G": user_request, "Recent_agent_G": "SpeechAgent"}

# Boss
def Boss_(state: GraphState):
    print("\n---BOSS---\n")

    start_time = time.time()

    request = state["Raw_user_request_G"]
    robot_current_position = state["Current_robot_position_G"]
    ai_response_json = Boss.process_user_request(request=request, robot_current_position=robot_current_position)

    Thought_boss = ai_response_json["Thought"]
    Next_agent_boss = ai_response_json["Action"]
    AgentInput_boss = ai_response_json["AgentInput"]

    print(f"next agent: {Next_agent_boss}")
    print(f"agent input: {AgentInput_boss}")
    
    end_time = time.time()
    time_function("Boss", start_time, end_time)

    return {"Thought_boss": Thought_boss, "Next_agent_boss": Next_agent_boss, "AgentInput_boss": AgentInput_boss, "Recent_agent_G": "Boss"}

# Speaker
def Speaker_(state: GraphState):
    print("\n---SPEAKER---\n")
    start_time = time.time()

    # check whether who calls the speaker
    Speaker_caller = state["Recent_agent_G"]

    # Speaker can be called by boss or workflow classifier or question answering agent
    if Speaker_caller == "Boss":
        Speaker_input = state["AgentInput_boss"]
    elif Speaker_caller == "WorkflowClassifier":
        Speaker_input = state["AgentInput_wc"]

    # get the user friendly response
    ai_response_json = Speaker.process_user_request(user_request=Speaker_input)
    message_to_user = str(ai_response_json["Action"])
    print(f"Message to user: {message_to_user}")
    # pass it to the speaker
    print("Speaking............")
    Speaker.speak(message_to_user)

    # Add the observation to the agents
    if Speaker_caller == "Boss":
        Boss.add_observations("Observation: Speaking completed")
    elif Speaker_caller == "WorkflowClassifier":
        WorkflowClassifier.add_observations("Observation: Speaking completed")

    end_time = time.time()
    time_function("Speaker", start_time, end_time)

    return {"Speaker_caller": Speaker_caller, "Output_speaker": message_to_user, "Recent_agent_G": "Speaker"}

# Top planner
def Top_planner(state: GraphState):
    print("\n---TOP PLANNER---\n")

    start_time = time.time()

    request = state["AgentInput_boss"]
    ai_response_json = TopPlanner.process_user_request(user_request=request)

    Thought_toppl = ai_response_json["Thought"]
    taskplan_toppl = ai_response_json["Action"]

    print(f"task plan: {taskplan_toppl}")
    
    end_time = time.time()
    time_function("TopPlanner", start_time, end_time)

    return {"Thought_toppl": Thought_toppl, "taskplan_toppl": taskplan_toppl, "Recent_agent_G": "TopPlanner"}

# Workflow classifier
def Workflow_classifier(state: GraphState):
    print("\n---WORKFLOW CLASSIFIER---\n")

    start_time = time.time()

    user_request = state["AgentInput_boss"]
    workflow_plan = state["taskplan_toppl"]
    # print(f"user request: {user_request}", f"\nworkflow plan: {workflow_plan}")
    ai_response_json = WorkflowClassifier.process_user_request(user_request=user_request, task_plan=workflow_plan)
    # print(f"ai_response_json: {ai_response_json}")

    Thought_wc = ai_response_json["Thought"]
    Next_agent_wc = ai_response_json["Action"]
    AgentInput_wc = ai_response_json["AgentInput"]

    print(f"Next agent: {Next_agent_wc}")
    print(f"Agent Input: {AgentInput_wc}")
    
    end_time = time.time()
    time_function("Workflow classifier", start_time, end_time)

    return {"Thought_wc": Thought_wc, "Next_agent_wc": Next_agent_wc, "AgentInput_wc": AgentInput_wc, "Recent_agent_G": "WorkflowClassifier"}

# Object searcher
def Object_searcher(state: GraphState):
    print("\n---OBJECT SEARCHER---\n")

    start_time = time.time()

    object_description = state["AgentInput_wc"]
    
    # Scanning for the object
    print(f"Start scanning....")
    max_scans = 8

    for i in range(max_scans):
        print(f"--- Scan attempt {i + 1} of {max_scans} ---")

        # scan for the object
        current_frame = Robot.get_current_RGB_camera_frames()
        response_json = ObjectSearcher.process_user_request(image=current_frame, object_label=object_description, draw_boxes=True)

        if response_json.get("Object_Found") == "found":
            print(f"Success! Object found on the scan")
            break

        # rotate
        Robot.rotate_to_scan_environment("left")

    if response_json.get("Object_Found") == "found":
        observation_object_searcher = "Found the object"
    else:
        observation_object_searcher = "Cannot found the object"

    # Add observation to the workflow classifier
    WorkflowClassifier.add_observations(f"Observation: {observation_object_searcher}")
    
    end_time = time.time()
    time_function("Object searcher", start_time, end_time)

    return {"Object_tobe_searched_os": object_description, "Object_search_status": observation_object_searcher, "Recent_agent_G": "ObjectSearcher"}

# Object grabber
def Object_grabber(state: GraphState):
    print("\n---OBJECT GRABBER---\n")

    start_time = time.time()

    object_description = state["AgentInput_wc"]
    # execute the object searcher here

    # Add observation to the workflow classifier
    observation_object_grabber = str(input("Enter the observation: "))
    WorkflowClassifier.add_observations(f"Observation: {observation_object_grabber}")
    
    end_time = time.time()
    time_function("Object grabber", start_time, end_time)

    return {"Recent_agent_G": "ObjectGrabber"}

# Question answerer
def Question_answerer(state: GraphState):
    print("\n---QUESTION ANSWERER---\n")

    start_time = time.time()

    question = state["AgentInput_wc"]
    user_current_position = state["Current_robot_position_G"]
    input_to_qa_agent = f"User question: {question} , User current position: {user_current_position}"

    print(f"Question from user: {question}")
    # execute the question answerer here
    ai_response_json = QuestionAnswerer.process_user_request(user_request=input_to_qa_agent)
    Explanation_qa = ai_response_json["Explanation"]
    FinalAnswer_qa = ai_response_json["FinalAnswer"]

    # Add observation to the workflow classifier
    WorkflowClassifier.add_observations(f"Observation: {FinalAnswer_qa}")
    
    end_time = time.time()
    time_function("Question answerer", start_time, end_time)

    return {"Explanation_qa": Explanation_qa, "FinalAnswer_qa": FinalAnswer_qa, "Recent_agent_G": "QuestionAnswerer"}

# Navigational supervisor main
def Navigation_supervisor_main(state: GraphState):
    print("\n---NAVIGATION SUPERVISOR MAIN---\n")
    start_time = time.time()

    # check whether who calls the navigation supervisor main
    if state["Recent_agent_G"] == "ErrorHandler":
        request = state["AgentInput_error"]
    else:
        request = state["AgentInput_wc"]

    ai_response_str = NavigationSupervisorMain.process_user_request(request=request)

    print(f"Navigation supervisor main: \n{ai_response_str}")

    # resetting all the chat histories
    NavigationSupervisorSec.reset_chat_messages()
    NavigationHandlerMain.reset_chat_messages()
    NavigationHandlerSec.reset_chat_messages()

    end_time = time.time()
    time_function("Navigation supervisor main", start_time, end_time)

    return {"response_navigation_supervisor_main": ai_response_str, "Recent_agent_G": "NavigationsupervisorMain"}

# Navigational supervisor secondary
def Navigation_supervisor_sec(state: GraphState):
    print("\n---NAVIGATION SUPERVISOR SEC---\n")

    start_time = time.time()

    task_plan_by_navsupmain = state["response_navigation_supervisor_main"]
    ai_response_json = NavigationSupervisorSec.process_user_request(request=task_plan_by_navsupmain)

    Thought_navsupsec = ai_response_json["Thought"]
    Action_navsupsec = ai_response_json["Action"]
    AgentInput_navsupsec = ai_response_json["AgentInput"]

    print(f"next agent: {Action_navsupsec}")
    print(f"agent input: {AgentInput_navsupsec}")
    
    end_time = time.time()
    time_function("Navigation supervisor secondary", start_time, end_time)

    return {"Thought_navsupsec": Thought_navsupsec, "Action_navsupsec": Action_navsupsec, "AgentInput_navsupsec": AgentInput_navsupsec, "Recent_agent_G": "NavigationsupervisorSec"}

# Final destination identifier
def Final_destination_identifier(state: GraphState):
    print("\n---FINAL DESTINATION IDENTIFIER---\n")

    start_time = time.time()

    agent_input = state["AgentInput_navsupsec"]
    response_by_final_destination_identifier_json = DestinationIdentifier.process_user_request(agent_input)

    destination_FDI = response_by_final_destination_identifier_json["Destination"]
    explanation_FDI = response_by_final_destination_identifier_json["Explanation"]

    formatted_response_by_FDI = f"Exact destination: {str(destination_FDI)}, Explanation: {str(explanation_FDI)}"
    response_by_FDI = f"Observation: {formatted_response_by_FDI}"

    # add response as observation
    NavigationSupervisorSec.add_observations(response_by_FDI)

    print(f"Final destination identifier: \n{formatted_response_by_FDI}")

    end_time = time.time()
    time_function("Final destination identifier", start_time, end_time)

    return {"Explanation_destination": formatted_response_by_FDI, "Recent_agent_G": "FinalDestinationIdentifier"}

# Current position identifier
def Current_position_identifier(state: GraphState):
    print("\n---CURRENT POSITION IDENTIFIER---\n")

    start_time = time.time()

    current_position = state["Current_robot_position_G"]
    current_position_str = f"Robot current position is {current_position}."
    response_by_CPI = f"Observation: {current_position_str}"

    # add response as observation
    NavigationSupervisorSec.add_observations(response_by_CPI)

    print(f"Current position identifier: \n{current_position_str}")

    end_time = time.time()
    time_function("Current position identifier", start_time, end_time)

    return {"Current_robot_position_CPI": current_position_str, "Recent_agent_G": "CurrentPositionIdentifier"}

# Waypoint generator
def Waypoint_generator(state: GraphState):
    print("\n---WAYPOINT GENERATOR---\n")

    start_time = time.time()

    agent_input = state["AgentInput_navsupsec"]
    response_by_waypointgenerator_json = WaypointGenerator.process_user_request(agent_input)

    waypoints_WG = response_by_waypointgenerator_json["WaypointSequence"]
    doorsequence_WG = response_by_waypointgenerator_json["doorSequence"]
    final_waypoints_WG = response_by_waypointgenerator_json["FinalSequence"]

    formatted_response_by_WG = f"Waypoints: {str(waypoints_WG)}, DoorSequence: {str(doorsequence_WG)}"
    response_by_WG = f"Observation: ({formatted_response_by_WG})"

    # add response as observation
    NavigationSupervisorSec.add_observations(response_by_WG)

    print(f"Waypoint generator: \n{formatted_response_by_WG}")

    end_time = time.time()
    time_function("Waypoint generator", start_time, end_time)

    return {"Room_waypoints": waypoints_WG, "Door_sequence": doorsequence_WG, "Final_waypoints": final_waypoints_WG, "Recent_agent_G": "WaypointGenerator"}

# Final navigational planner
def Final_navigational_planner(state: GraphState):
    print("\n---FINAL NAVIGATIONAL PLANNER---\n")

    start_time = time.time()

    agent_input = state["AgentInput_navsupsec"]
    response_by_FNP_json = FinalNavigationPlanner.process_user_request(agent_input)

    plan_FNP_str = response_by_FNP_json["The_plan"]

    print(f"Final Navigation planner: \n{response_by_FNP_json}")

    # reset chat history
    NavigationHandlerMain.reset_chat_messages()

    end_time = time.time()
    time_function("Final navigational planner", start_time, end_time)

    return {"Plan_FNP": plan_FNP_str, "Recent_agent_G": "FinalNavigationalPlanner"}

# Navigation handler main
def Navigation_handler_main(state: GraphState):
    print("\n---NAVIGATION HANDLER MAIN---\n")

    start_time = time.time()

    agent_input = state["Plan_FNP"]
    next_task_NHM_json = NavigationHandlerMain.process_user_request(agent_input)

    Thought_NHM = next_task_NHM_json["Thought"]
    Action_NHM = next_task_NHM_json["Action"]
    Action_type_NHM = next_task_NHM_json["ActionType"]

    next_task_NHM_str = f"Thought: {Thought_NHM}, Action: {Action_NHM}"

    print(f"Navigation handler main: \n{next_task_NHM_str}")

    # reset chat history
    NavigationHandlerSec.reset_chat_messages()
    end_time = time.time()
    time_function("Navigation handler main", start_time, end_time)

    return {"Next_task_NHM": next_task_NHM_str, "Action_type_NHM": Action_type_NHM, "Recent_agent_G": "NavigationHandlerMain"}

# Navigation handler secondary
def Navigation_handler_sec(state: GraphState):
    print("\n---NAVIGATION HANDLER SEC---\n")

    start_time = time.time()

    agent_input = state["Next_task_NHM"]
    ai_response_json = NavigationHandlerSec.process_user_request(request=agent_input)

    Thought_NHS = ai_response_json["Thought"]
    Action_NHS = ai_response_json["Action"]
    AgentInput_NHS = ai_response_json["AgentInput"]

    print(f"next agent: {Action_NHS}")
    print(f"agent input: {AgentInput_NHS}")

    # if the next agent is the navigation handler main, a completion message should be added into the chat history of it.
    if Action_NHS == "NavigationHandlerMain":
        response_by_NHS_formatted = f"Observation: {AgentInput_NHS}"
        NavigationHandlerMain.add_observations(response_by_NHS_formatted)

    end_time = time.time()
    time_function("Navigation handler secondary", start_time, end_time)

    return {"Thought_navhansec": Thought_NHS, "Action_navhansec": Action_NHS, "AgentInput_navhansec": AgentInput_NHS, "Recent_agent_G": "NavigationHandlerSec"}

# Door checker
def Door_checker(state: GraphState):
    print("\n---DOOR CHECKER---\n")

    start_time = time.time()

    agent_input = Robot.get_current_RGB_camera_frames()
   
    response_by_DC_json = DoorStatusChecker.process_user_request(agent_input)

    door_detected = response_by_DC_json["Door_label"]
    door_status = response_by_DC_json["Door_status"]
    door_status_explanation = response_by_DC_json["Explanation"]

    print(f"Door checker: {door_status_explanation}")

    response_by_DC_formatted = f"Observation: {door_status_explanation}"

    # add response as observation
    NavigationHandlerSec.add_observations(response_by_DC_formatted)

    end_time = time.time()
    time_function("Door checker", start_time, end_time)

    return {"Door_status_explanation": door_status_explanation, "Recent_agent_G": "DoorChecker"}

# Path planner
def Path_planner(state: GraphState):
    print("\n---PATH PLANNER---\n")

    start_time = time.time()

    robot_target_position = state["AgentInput_navhansec"]
    robot_current_position = state["Current_robot_position_G"]

    agent_input = f"Robot current position: {robot_current_position}, robot target position: {robot_target_position}"
    ai_response_json = PathPlanner.process_user_request(agent_input)

    robot_current_coord = ai_response_json["robot_current_pos_coord"]
    robot_target_coord = ai_response_json["robot_target_pos_coord"]
    encountered_door_num = ai_response_json["door_num"]
    robot_target_orientation = ai_response_json["target_orientation"]
    robot_target_position_ = ai_response_json["target_position"]

    # Short-circuit when the robot is already at the requested waypoint.
    # Why: FinalNavigationalPlanner sometimes emits a redundant first step like
    # "Go to R3" when the robot is already in R3. Without this guard we'd call
    # update_binary_map with the fake door_num=1 fallback and generate_path to
    # the current position, which corrupts the map and stalls the graph.
    if robot_current_coord == robot_target_coord:
        response_by_PP_str = (
            f"Robot is already at {robot_target_position_}. "
            "Step complete; no movement needed."
        )
        print(f"Path planner: {response_by_PP_str}")
        NavigationHandlerSec.add_observations(f"Observation: {response_by_PP_str}")

        end_time = time.time()
        time_function("Path planner", start_time, end_time)

        return {
            "Path_planning_status": response_by_PP_str,
            "Target_orientation": robot_target_orientation,
            "Target_position": robot_target_position_,
            "Recent_agent_G": "PathPlanner",
        }

    # update the door in the map and plan the map
    print(f"[Allinone.Path_planner] calling update_binary_map door_num={encountered_door_num}", flush=True)
    Robot.update_binary_map(door_num=encountered_door_num, status=True)
    print(f"[Allinone.Path_planner] update_binary_map returned; calling generate_path target={robot_target_coord}", flush=True)
    success = Robot.generate_path(target_position=robot_target_coord, door_num=encountered_door_num, show_rt_path=False)
    print(f"[Allinone.Path_planner] generate_path returned success={success}", flush=True)

    if success:
        response_by_PP_str = "Path planning done succesfully."
    else:
        response_by_PP_str = "Path planning failed. Try again."

    print(f"Path planner: {response_by_PP_str}")

    response_by_PP_formatted = f"Observation: {response_by_PP_str}"

    # add response as observation
    NavigationHandlerSec.add_observations(response_by_PP_formatted)

    end_time = time.time()
    time_function("Path planner", start_time, end_time)

    return {"Path_planning_status": response_by_PP_str, "Target_orientation": robot_target_orientation, "Target_position": robot_target_position_, "Recent_agent_G": "PathPlanner"}

# Robot executor
def Robot_executor(state: GraphState):
    print("\n---ROBOT EXECUTOR---\n")

    start_time = time.time()

    target_orientation = state["Target_orientation"]
    robot_target_position = state["Target_position"]
    robot_current_position = state["Current_robot_position_G"]

    # execute the robot
    Robot.Execute_robot(target_orientation=target_orientation)
    while True:
        time.sleep(1)

        if Robot.successful_robot_movement == True:
            break

    print("robot reached the position")
    
    if Robot.successful_robot_movement == True:
        response_by_RE_str = "Robot executed completed"
        if robot_target_position[0] == "R" or robot_target_position[0] == "C":
            robot_current_position = robot_target_position

    print(f"Robot executor: {response_by_RE_str}")

    response_by_RE_formatted = f"Observation: {response_by_RE_str}"

    # add response as observation
    NavigationHandlerSec.add_observations(response_by_RE_formatted)

    end_time = time.time()
    time_function("Robot executor", start_time, end_time)

    return {"Robot_executor_status": response_by_RE_str, "Current_robot_position_G": robot_current_position, "Recent_agent_G": "RobotExecutor"}

# Finisher
def Finisher(state: GraphState):
    print("---FINISHER---\n")

    df = pd.DataFrame(timing_data)
    df.to_excel(r"D:\Tensorflow learning\Robot tesing\test10_mas\execution_times.xlsx", index=False)

    print("Execution times saved to 'execution_times.xlsx'")

    print("resetting history....")

    # Boss agent
    Boss.reset_chat_messages()
    # Speaker agent no need
    # top planner
    TopPlanner.reset_chat_messages()
    # workflow classifier
    WorkflowClassifier.reset_chat_messages()
    # object searcher no need
    # object grabber no need
    # question answerer no need
    # navigation supervisor main
    NavigationSupervisorMain.reset_chat_messages()
    # navigation supervisor secondary
    NavigationSupervisorSec.reset_chat_messages()
    # final destination identifier no need
    # current position idetifier no need
    # waypoint generator no need
    # final navigational planner no need
    # navigational handler main
    NavigationHandlerMain.reset_chat_messages()
    # navigational handler secondary
    NavigationHandlerSec.reset_chat_messages()
    # door checker no need
    # path planner no need
    # robot executor no need
    # error handler
    ErrorHandler.reset_chat_messages()

    # reset timing data
    reset_timing_data()

    print("---DONE---")

    # resetting the graph state except Current_robot_position_G
    return {
            # general parameters
            "Current_robot_position_G": state["Current_robot_position_G"],
            "Recent_agent_G": "",
            "Raw_user_request_G": "",
            "Transcriptioner_status": state["Transcriptioner_status"],

            # parameters of the boss
            "Thought_boss": "",
            "Next_agent_boss": "",
            "AgentInput_boss": "",

            # parameters of the speaker
            "Speaker_caller": "",
            "Output_speaker": "",

            # parameters of the top planner
            "Thought_toppl": "",
            "taskplan_toppl": "",

            # parameters of the workflow classifier
            "Thought_wc": "",
            "Next_agent_wc": "",
            "AgentInput_wc": "",

            # parameters of the object searcher
            "Object_tobe_searched_os": "",
            "Object_search_status": "",
            "Object_location_os": "",

            # parameters of the object grabber
            "Object_tobe_grabbed_og": "",
            "Object_location_og": "",
            "Object_grabber_status": "",

            # parameters of the question answerer
            "Explanation_qa": "",
            "FinalAnswer_qa": "",

            # parameters of the navigational supervisor main
            "response_navigation_supervisor_main": "",

            # parameters of the navigational supervisor sec
            "Thought_navsupsec": "",
            "Action_navsupsec": "",
            "AgentInput_navsupsec": "",

            # parameters of destination decisioner
            "Explanation_destination": "",

            # parameters of current position identifier
            "Current_robot_position_CPI": "",

            # parameters of waypoint generator
            "Room_waypoints": "",
            "Door_sequence": "",
            "Final_waypoints": "",

            # parameters of final navigational planner
            "Plan_FNP": "",

            # parameters of navigation handler main
            "Next_task_NHM": "",
            "Action_type_NHM": "",

            # parameters of navigation handler secondary
            "Thought_navhansec": "",
            "Action_navhansec": "",
            "AgentInput_navhansec": "",

            # parameters of door checker
            "Door_status_explanation": "",

            # parameters of path planner
            "Path_planning_status": "",
            "Target_orientation": "",
            "Target_position": "",

            # parameters of robot executor
            "Robot_executor_status": "",

            # parameters of error handler
            "Thought_error": "",
            "Action_error": "",
            "AgentInput_error": "",

            # parameters of the finisher
            "check_finisher": ""
        }

# Error handler
def Error_handler(state: GraphState):
    print("\n---ERROR_HANDLER---\n")

    Error_msg_sender = state["Recent_agent_G"]

    # identify the error handler
    if Error_msg_sender == "NavigationsupervisorSec":
        Error_occured = state["AgentInput_navsupsec"]
    elif Error_msg_sender == "NavigationHandlerSec":
        Error_occured = state["AgentInput_navhansec"]
    else:
        Error_occured = None

    print(f"Error occurred: {Error_occured}")

    ai_response_json = ErrorHandler.process_user_request(Error_occured)

    print(f"ai_response_json: {ai_response_json}")

    Thought_error = ai_response_json["Thought"]
    Next_agent_error = ai_response_json["Action"]
    AgentInput_error = ai_response_json["AgentInput"]

    print(f"next_agent_error: {Next_agent_error}")

    # router for the erros
    if Next_agent_error == "NavigationsupervisorMain":
        print("\n\nBack to navigation supervisor main\n\n")
        return {"Thought_error": Thought_error, "Action_error": Next_agent_error, "AgentInput_error": AgentInput_error, "Recent_agent_G": "ErrorHandler"}
    elif Next_agent_error == "WorkflowClassifier":
        print("\n\nBack to workflow classifier\n\n")
        observation_error = AgentInput_error
        WorkflowClassifier.add_observations(f"Observation: {observation_error}")

        return {"Thought_error": Thought_error, "Action_error": Next_agent_error, "AgentInput_error": AgentInput_error, "Recent_agent_G": "ErrorHandler"}

# Router function Boss
def Router_boss(state: GraphState):
    next_agent = state["Next_agent_boss"]
    return next_agent

# Router function Speaker
def Router_speaker(state: GraphState):
    next_agent = state["Speaker_caller"]
    return next_agent

# Router function Workflow classifier
def Router_workflow_classifier(state: GraphState):
    next_agent = state["Next_agent_wc"]

    if next_agent == "NavigationsupervisorMain":
        NavigationSupervisorMain.reset_chat_messages()

    return next_agent

# Router function Navigation supervisor secondary
def Router_navsupsec(state: GraphState):
    next_agent = state["Action_navsupsec"]
    return next_agent

# Router function Navigation handler main
def Router_navhanmain(state: GraphState):
    action_type = state["Action_type_NHM"]

    if action_type == "Continue":
        return "NavigationHandlerSec"
    elif action_type == "Finish":
        observation = "Completed the navigation"
        WorkflowClassifier.add_observations(f"Observation: {observation}")
        return "WorkflowClassifier"

# Router function Navigation handler secondary
def Router_navhansec(state: GraphState):
    next_agent = state["Action_navhansec"]
    return next_agent

# Router function Error handler
def Router_errorhandler(state: GraphState):
    next_agent = state["Action_error"]
    return next_agent

# LAN-GRAPH
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("SpeechAgent", Speech_agent)
workflow.add_node("Boss", Boss_)
workflow.add_node("Speaker", Speaker_)
workflow.add_node("TopPlanner", Top_planner)
workflow.add_node("WorkflowClassifier", Workflow_classifier)
workflow.add_node("ObjectSearcher", Object_searcher)
workflow.add_node("ObjectGrabber", Object_grabber)
workflow.add_node("QuestionAnswerer", Question_answerer)
workflow.add_node("NavigationsupervisorMain", Navigation_supervisor_main)
workflow.add_node("NavigationsupervisorSec", Navigation_supervisor_sec)
workflow.add_node("FinalDestinatioIdentifier", Final_destination_identifier)
workflow.add_node("CurrentPositionIdentifier", Current_position_identifier)
workflow.add_node("WaypointGenerator", Waypoint_generator)
workflow.add_node("FinalNavigationalPlanner", Final_navigational_planner)
workflow.add_node("NavigationHandlerMain", Navigation_handler_main)
workflow.add_node("NavigationHandlerSec", Navigation_handler_sec)
workflow.add_node("DoorChecker", Door_checker)
workflow.add_node("PathPlanner", Path_planner)
workflow.add_node("RobotExecutor", Robot_executor)
workflow.add_node("Finisher", Finisher)
workflow.add_node("ErrorHandler", Error_handler)

# Build graph
workflow.add_edge(START, "SpeechAgent")
workflow.add_edge("SpeechAgent", "Boss")
workflow.add_conditional_edges (
                                    "Boss",
                                    Router_boss,
                                    {
                                        "TopPlanner": "TopPlanner",
                                        "Speaker": "Speaker",
                                        "Finisher": "Finisher",
                                    }
                                )
workflow.add_conditional_edges (
                                    "Speaker",
                                    Router_speaker,
                                    {
                                        "Boss": "Boss",
                                        "WorkflowClassifier": "WorkflowClassifier",
                                    }
                                )
workflow.add_edge("TopPlanner", "WorkflowClassifier")
workflow.add_conditional_edges (
                                    "WorkflowClassifier",
                                    Router_workflow_classifier,
                                    {
                                        "NavigationsupervisorMain": "NavigationsupervisorMain",
                                        "ObjectSearcher": "ObjectSearcher",
                                        "ObjectGrabber": "ObjectGrabber",
                                        "Speaker": "Speaker",
                                        "Finisher": "Finisher",
                                        "QuestionAnswerer": "QuestionAnswerer",
                                    }
                                )
workflow.add_edge("ObjectSearcher", "WorkflowClassifier")
workflow.add_edge("ObjectGrabber", "WorkflowClassifier")
workflow.add_edge("QuestionAnswerer", "WorkflowClassifier")
# workflow.add_edge("NavigationsupervisorMain", END)
workflow.add_edge("NavigationsupervisorMain", "NavigationsupervisorSec")
workflow.add_conditional_edges (
                                    "NavigationsupervisorSec",
                                    Router_navsupsec,
                                    {
                                        "FinalDestinatioIdentifier": "FinalDestinatioIdentifier", 
                                        "CurrentPositionIdentifier": "CurrentPositionIdentifier",
                                        "WaypointGenerator": "WaypointGenerator",
                                        "FinalNavigationalPlanner": "FinalNavigationalPlanner",
                                        "ErrorHandler": "ErrorHandler",
                                    }
                                )
workflow.add_edge("FinalDestinatioIdentifier", "NavigationsupervisorSec")
workflow.add_edge("CurrentPositionIdentifier", "NavigationsupervisorSec")
workflow.add_edge("WaypointGenerator", "NavigationsupervisorSec")

workflow.add_edge("FinalNavigationalPlanner", "NavigationHandlerMain")
workflow.add_conditional_edges (
                                    "NavigationHandlerMain",
                                    Router_navhanmain,
                                    {
                                        "NavigationHandlerSec": "NavigationHandlerSec",
                                        "WorkflowClassifier": "WorkflowClassifier",
                                    }
                                )

workflow.add_conditional_edges (
                                    "NavigationHandlerSec",
                                    Router_navhansec,
                                    {
                                        "NavigationHandlerMain": "NavigationHandlerMain",
                                        "DoorChecker": "DoorChecker",
                                        "PathPlanner": "PathPlanner",
                                        "RobotExecutor": "RobotExecutor",
                                        "ErrorHandler": "ErrorHandler",
                                    }
                                )

workflow.add_edge("DoorChecker", "NavigationHandlerSec")
workflow.add_edge("PathPlanner", "NavigationHandlerSec")
workflow.add_edge("RobotExecutor", "NavigationHandlerSec")

workflow.add_conditional_edges (
                                    "ErrorHandler",
                                    Router_errorhandler,
                                    {
                                        "NavigationsupervisorMain": "NavigationsupervisorMain", 
                                        "WorkflowClassifier": "WorkflowClassifier",
                                    }
                                )

workflow.add_edge("Finisher", "SpeechAgent")

app = workflow.compile()

app.invoke({"Current_robot_position_G": 'R3', "Transcriptioner_status": "on"}, config={"recursion_limit": 1000})
