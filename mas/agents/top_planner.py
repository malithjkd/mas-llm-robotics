"""Top Planner Agent"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class output_struct_topplanner(BaseModel):
    Thought: str = Field(description="you should always think the planning process step by step")
    Action: str = Field(description="Whole action plan with the reasoning step by step")

class TopPlannerAgent():
    """
    This agent is responsible for get the instructions from the user and plan the most highlevel 
    plan according to the given few shot prompt.
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
        self.chat_messages = []

    def load_additional_info(self):
        self.base_path = os.path.dirname(__file__)
        
        with open(os.path.join(self.base_path, "Top_planner_supporting_info.txt"), 'r') as file:
            self.supportive_info = file.read()
        
    def load_prompts(self):
        # Set up the agent prompt response
        self.Top_Planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a intelligent planner who is responsible on creating action plans for a robot to execute a given task.
                - User request: {user_request}.\n
                - Rules and supportive info: {supportive_info}\n.
                Please follow these set of rules strictly when you give your response:
                - User request includes request from user, request type and also the robot current position.
                - Decide whether the request is a Question or a Navigational task according to the given details.
                - Then you have to plan the high level task plan according to that.
                - When you plan the task plan, follow the examples and rules in the supportive info.
                - Strictly follow the following format when you give your response: {format_instructions}\n
                """),
                (
                "system",
                """
                Make sure you create the whole action plan.
                Please make sure the robot is strickly limited to the following functions:
                1. Navigate to a place
                2. Search object
                3. Grab object
                4. Question and answering
                Don't add any additional functions to the robot.
                """,
                ),
            ]
        )
    
    def setup_llm_chains(self):
        """Chain 1: LLM for response"""
        llm_for_response = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=output_struct_topplanner)
        self.final_response_chain  = self.Top_Planner_prompt | llm_for_response | self.parser_output

    def process_user_request(self, user_request):
        user_request = user_request

        formatted_json_response = self.final_response_chain.invoke({"user_request": user_request,
                                                                    "supportive_info": self.supportive_info,
                                                                    "format_instructions": self.parser_output.get_format_instructions()})

        return formatted_json_response

    def reset_chat_messages(self):
        self.chat_messages = []

# if __name__ == "__main__":
#     dd_agent = TopPlannerAgent()

#     user_inputs = [ "The robot current position is R1, user request is to find and retrieve a screw driver, this is a navigational command.",
#                   "User request is 'I need a screw driver', robot current position is R1, this is a navigational task", 
#                   "User request is Can you take me to the KUKA robot, robot current position is R1, navgational task",
#                   "user request is Can you tell me how can I go to the KUKA robot, robot current position is R1, this is a question",
#                   "user request is 'I need a bottle of water', robot current position is R1, this is a navigational task",
#                   ]

#     for user_input in user_inputs:
#         ai_response = dd_agent.process_user_request(user_input)
#         print(f"User Input: {user_input}")
#         print(f"AI: {ai_response}")
#         print()
