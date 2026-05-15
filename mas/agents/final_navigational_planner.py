"""Final Navigation planner agent"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage

class FinalNavigationalPlannerAgent_output_struct(BaseModel):
    Explanation: str = Field(description="Explanation about how you decide the plan step by step.")
    The_plan: str = Field(description="The navigational plan which you have created step by step. Make sure to follow the exact format in the examples and Do not add any other details.")

class FinalNavigationalPlannerAgent():
    """This agent is responsible for create the navigational plan."""

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
        
        with open(os.path.join(self.base_path, "Example_scenarios_plan.txt"), 'r') as file:
            self.example_plan_info = file.read()

        with open(os.path.join(self.base_path, "Example_final_waypoints.txt"), 'r') as file:
            self.exmaple_waypoints_info = file.read()

    def load_prompts(self):
        self.waypoint_modifier = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a robot navigational planner who has the responsibility on creating the navigational plan by combining waypoints with the door sequences to create the final waypoint sequence.
                The details and waypoints : {user_details}.\n
                - In the details given to you, includes the robot current position, robot destination position, waypoints and door sequences.
                - Concatanate Door sequence with the waypoint sequence.
                - The order should be like this: '1st waypoint in the waypoint sequence->1st door in the door sequence->2nd waypoint->second door.........'
                - Examples scenarios: {example_waypoint_scenarios}. \n
                - If the door sequnce is not provided, don't put the doors just continue with the given waypoints.
                - Your final response should strickly follow the following format:
                    -- Thought: <You should think step by step how you are going to create the final waypoint sequence.>
                    -- Final output: <This should include these details: robot current position, destination position and the final waypoint sequence.>
                """),
            ]
        )

        # Set up the agent prompt
        self.Navigational_supervisor_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a robot Highlevel navigational planner who has the responsibility on creating the highlevel navigational plan according to the waypoints and details given by the user.
                Please follow these rules when you give a response:
                - The user details includes the robot current position, robot destination position and the waypoint sequence.
                - Waypoints include the sequence of places and doors in orderly manner which robot have to pass to reach the destination.
                - Get those details and create a highlevel navigational plan according to the given waypoint order.
                - In the plan you can include the below functions.
                    - go from one place to another.
                    - If you encounter a door, you should check whether the door is open or not.
                - You are provided with some of the example scenariors. Examples scenarios: {example_plan_scenarios}. \n
                - Make sure you follow the ***exact format which is given in the examples***, when you are creating the plan.
                - Your final response should strickly follow the following format: {format_instructions}.\n
                - When you plan the map, make sure you use all the waypoints including the current position of the robot.
                """),
                MessagesPlaceholder(variable_name="user_details"),
            ]
        )
    
    def setup_llm_chains(self):
        """Chain 1: LLM for waypoint formatter"""
        llm_formatter = ChatOpenAI(temperature=0.1, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.chain_formatter  = self.waypoint_modifier | llm_formatter

        """Chain 2: LLM for highlevel planner"""
        llm_planner = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=FinalNavigationalPlannerAgent_output_struct)
        self.chain_planner  = self.Navigational_supervisor_prompt | llm_planner | self.parser_output

    def process_user_request(self, request):
        self.user_details = request
        formated_response = self.chain_formatter.invoke({"user_details": self.user_details,
                                                        "example_waypoint_scenarios": self.exmaple_waypoints_info})
        formated_response = formated_response.content
        formated_response = [HumanMessage(content=formated_response)]

        plan_response_json = self.chain_planner.invoke({"user_details": formated_response,
                                                        "example_plan_scenarios": self.example_plan_info, 
                                                        "format_instructions": self.parser_output.get_format_instructions()})
        
        return plan_response_json

# if __name__ == "__main__":
#     dd_agent = FinalNavigationalPlannerAgent()

#     test_cases = [
#                 "Robot currently in R1, Robot need to go to R3, The waypoints are R1->R2->R3, The door sequence is D2->D3",
#                 "Robot currently in R3, Robot need to go to R1, The waypoints are R3->R2->R1, The door sequence is D3->D2",
#                 "Robot currently in R1, Robot need to go to C, The waypoints are R1->R2->R3->C, The door sequence is D2->D3->D4",
#                 "Robot currently in C, Robot need to go to R2, The waypoints are C->R3->R2, The door sequence is D4->D3",
#                 "Robot currently in C, Robot need to go to R1, The waypoints are C->R3->R2->R1, The door sequence is D4->D3->D2",
#                 "Robot currently in C, Robot need to come back to the C again by going a round, The waypoints are C->R3->R2->R1->C, The door sequence is D4->D3->D2->D1",
#                 ]
    
#     for x in test_cases:
#         response = dd_agent.process_user_request(x)
#         print(f"User: {x}")
#         print(f"Response: {response}")