"""Navigation Handler Main"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class NavigationalHandlerMain_output_struct(BaseModel):
    Thought: str = Field(description="you should always think step by step to determine what to do next according to the previous observation and the given task plan")
    Action: str = Field(description="the action should be taken next")
    ActionType: str = Field(description="""This should be one of 'Continue' or 'Finish' to indicate whether the plan executing is finish or not.
                                        If your action is a task, then ActionType should be 'Continue'""")

class NavigationalHandlerMainAgent():
    """
    This agent is responsible for giving the relevant commands to the next agent to 
    fullfill the navigational task.
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

    def load_prompts(self):
        # Set up the agent prompt
        self.Navigational_supervisor_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a supervisor who is responsible on executing a given navigational plan step by step.
                The task plan is: {action_plan}.\n
                Please follow these rules when you give a response:
                - The task plan should execute step by step.
                - Refer the conversation history carefully and decide the next task to do.
                - After each task call, you will be provided whether the task is done or not as the observation.\n
                """),
                MessagesPlaceholder(variable_name="messages"),
                (
                "system",
                """
                Given the conversation above, you should decide the next task according to the task plan and the observations.
                - Make sure you give the next task ***as it is given in the task plan***. Do not change the task.
                Make sure you take ***one action at a time***. You need the observation before going to the next action.
                - Strictly follow the following format when you give your response:
                    -- {format_instructions}.\n
                    ... (this Thought/Action/Action Type/Observation can repeat N times. Here observation will be given by the user.)
                If you successfully finish all the tasks,
                - Action should be 'Finish' and ActionType should be 'Finish'.
                Make sure you ***follow the given format.***
                """,
                ),
            ]
        )
    
    def setup_llm_chains(self):
        """Chain 1: LLM for response"""
        llm_for_outputparser = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=NavigationalHandlerMain_output_struct)
        self.final_output_chain  = self.Navigational_supervisor_prompt | llm_for_outputparser | self.parser_output

    def process_user_request(self, request):
        self.task_plan = request
        response_json = self.final_output_chain.invoke({"messages": self.chat_messages,
                                                       "action_plan": self.task_plan, 
                                                       "format_instructions": self.parser_output.get_format_instructions()})
        
        Thought_NHM = response_json["Thought"]
        Action_NHM = response_json["Action"]
        Action_type_NHM = response_json["ActionType"]

        formatted_msg = f"Next task: {Action_NHM}"
        self.chat_messages.append(AIMessage(content=formatted_msg))

        return response_json

    def add_observations(self, observation):
        observation = observation
        self.chat_messages.append(HumanMessage(content=observation))

    def reset_chat_messages(self):
        self.chat_messages = []

# if __name__ == "__main__":
#     dd_agent = NavigationalHandlerMainAgent()
#     task_plan = """
#                 -- The plan: 
#                 1. Go to R1.
#                 2. Go to D2.
#                 3. Check D2 is open or closed.
#                 4. If D2 open, go to R2.
#                 5. Go to D3.
#                 6. Check D3 is open or closed.
#                 7. If D3 open, go to R3.

#                 -- Explanation: 
#                 1. The robot starts in R1, so the first step is to confirm its current position.
#                 2. The next waypoint is D2, so the robot moves to D2.
#                 3. At D2, the robot must check if the door is open or closed.
#                 4. If D2 is open, the robot proceeds to R2, the next waypoint.
#                 5. The robot then moves to D3, the next door in the sequence.
#                 6. At D3, the robot checks if the door is open or closed.
#                 7. If D3 is open, the robot proceeds to R3, the final destination.
#                 """

#     while True:
#         response = dd_agent.process_user_request(task_plan)
#         print(f"response: {response}")
#         observation = input("Observation: ")
#         formatted_observation = f"Observation: {observation}"
#         dd_agent.add_observations(formatted_observation)


