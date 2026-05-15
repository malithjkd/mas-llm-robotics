"""Path planner"""
import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional

class output_struct_PathPlannerAgent(BaseModel):
    Thought: str = Field(description="you should always think step by step.")
    Current_position: str = Field(description="Current position of the robot. This should be one of 'R1, R2, R3, C'")
    Target_position: str = Field(description="Target position of the robot. This should be one of 'R1, R2, R3, C, D1, D2, D3, D4'")
    Door_encounter: Optional[int] = Field(description="""   The door number which robot encounting when it is moving to the destination position through the generated path.
                                                            If it is not encounting any doors, give None as the door number.
                                                            Make sure you give only the ***door number*** or None.""")

class PathPlannerAgent():
    """
    This agent is responsible for plan the path.
    """
    def __init__(self):
        # load the api keys
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.langchain_tracking_v2 = os.getenv("LANGCHAIN_TRACING_V2")
        self.langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT")

        # load the parameter base
        self.load_additional_info()
        # load the prompts
        self.load_prompts()
        # Set up the LLM
        self.setup_llm_chains()

    def load_additional_info(self):
        self.base_path = os.path.dirname(__file__)
        # info about agents
        with open(os.path.join(self.base_path, "Environment_description_for_PP.txt"), 'r') as file:
            self.env_info = file.read()

        self.map_location_coords = {
                                    "CD1": ((150, 1100), -145),
                                    "R1D1": ((165, 645), -30),
                                    "R1D2": ((600, 465), 90),
                                    "R2D2": ((1125, 465), -90),
                                    "R2D3": ((1275, 465), 90),
                                    "R3D3": ((1745, 465), -90),
                                    "R3D4": ((1560, 715), 0),
                                    "CD4": ((1470, 1100), 145),

                                    "R1": ((195, 465), None),
                                    "R2": ((1200, 465), None),
                                    "R3": ((1845, 480), None),
                                    "C": ((1200, 1065), None),

                                    "R1R1": ((195, 465), None),
                                    "R2R2": ((1200, 465), None),
                                    "R3R3": ((1845, 480), None),
                                    "CC": ((1200, 1065), None),

                                    "R1R2": ((1200, 465), 90),
                                    "R1C": ((1200, 1065), 90),
                                    "R2R1": ((195, 465), -90),
                                    "R2R3": ((1845, 480), 90),
                                    "R3R2": ((1200, 465), -90),
                                    "R3C": ((1200, 1065), -90),
                                    "CR3": ((1845, 480), 180),
                                    "CR1": ((195, 465), 180),
                                }


    def load_prompts(self):
        # Set up the agent prompt
        self.Path_planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are an intelligent map reader who has the capability on understanding the given maps to extract the required information.
                - The map and the supportive information: {env_info}.\n
                - User details: {user_details}.\n
                Your task is to extract the current robot position, destination position and the doors which are going to encounter through the path.
                - First Refer the user details to identify the current position and destination position of the robot.
                - Then, refer the map carefully to identify the path from the current position to the destination position.
                - Carefully refer the generated path in the previous step to identify the doors which are going to encounter.
                When you give the response, Make sure you strickly follow the following format: {format_instructions}.\n
                """),
            ]
        )
    
    def setup_llm_chains(self):
        """Chain 1: LLM for output parser"""
        llm_for_outputparser = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=output_struct_PathPlannerAgent)
        self.final_output_chain  = self.Path_planner_prompt | llm_for_outputparser | self.parser_output

    def process_user_request(self, request):
        self.user_details = request

        response_json = self.final_output_chain.invoke({"user_details": self.user_details,
                                                        "env_info": self.env_info,
                                                        "format_instructions": self.parser_output.get_format_instructions()})
        
        print("Response JSON:\n", response_json)
        
        self.current_robot_position = response_json["Current_position"]
        self.target_robot_position = response_json["Target_position"]
        self.door_robot_encounter = response_json["Door_encounter"]

        prefix = self.current_robot_position + self.target_robot_position

        robot_target_pos_coord = self.map_location_coords[prefix][0]
        robot_current_pos_coord = self.map_location_coords[self.current_robot_position][0]
        robot_target_orientation = self.map_location_coords[prefix][1]

        if self.door_robot_encounter == None or self.door_robot_encounter == "None":
            self.door_robot_encounter = 1

        response_formatted_json = {
                                    "robot_current_pos_coord": robot_current_pos_coord, 
                                    "robot_target_pos_coord": robot_target_pos_coord, 
                                    "door_num": self.door_robot_encounter,
                                    "target_orientation": robot_target_orientation,
                                    "target_position": self.target_robot_position,
                                  }

        return response_formatted_json

# if __name__ == "__main__":
#     dd_agent = PathPlannerAgent()
#     use_cases = [
#         "The robot is currently at R1 and the target position is R2", 
#         "The robot is currently at R2 and the target position is R3",
#         "The robot is currently at R3 and the target position is C",
#         "The robot is currently at C and the target position is R1",
#         "The robot is currently at R1 and the target position is D1",
#     ]

#     for use_case in use_cases:
#         print(f"Use Case: {use_case}")
#         response = dd_agent.process_user_request(use_case)
#         print("Response:", response)
#         print("\n", "-" * 50, "\n")
