"""Navigation Handler Secondary"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class output_struct_NavigationalHandlerSecAgent(BaseModel):
    Thought: str = Field(description="you should always think step by step to determine what to do next according to the previous observation and the given task")
    Action: str = Field(description="The name of the next agent to activate")
    AgentInput: str = Field(description="This should be the input arguments to the agent separated by commas")

class NavigationalHandlerSecAgent():
    """
    This agent is responsible for continue the conversation between the agents under itself.
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
        # member agents
        self.agent_members = ["DoorChecker", "PathPlanner", "RobotExecutor"]
        # info about agents
        with open(os.path.join(self.base_path, "Agent_description_navigation_NHS.txt"), 'r') as file:
            self.agent_info = file.read()
        # special notes about agents
        with open(os.path.join(self.base_path, "Agent_NHS_special_notes.txt"), 'r') as file:
            self.special_info = file.read()

    def load_prompts(self):
        # Set up the agent prompt
        self.Navigational_supervisor_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a supervisor who is responsible on continue the conversation between agents under yourself to executing a given task on navigating the robot.
                The given task is: {given_task}. Here 'D' refers to doors, 'R' refers to rooms and 'C' refers to corridor\n
                Please follow these rules when you give a response:
                - The agents works under you are : {agent_members}.\n
                - The details about agents: {agent_info}.\n
                - Special info about agents: {special_info}.\n
                - The task you have to do is, refer the conversation history and decide which agent to activate next and its inputs.
                - Make sure you decide the right agent with the correct input arguments.
                - After you execute that agent, you will be given the results and observations of that agent.
                - According to the observation, you can pick which agent to activate next. The observation will be an error or completion message.
                """),
                MessagesPlaceholder(variable_name="messages"),
                (
                "system",
                """
                Given the conversation above, you should decide the next agent according to the task and the obseravtions.
                Make sure you take ***one action at a time***. You need the observation before going to the next action.
                - Strictly follow the following format when you give your response:\n
                    --{format_instructions}.\n
                    ... (this Thought/Action/AgentInputs/Observation can repeat N times. Here observation will be given by the next agent after it finish the work.)
                If the observation is mentioned as an error, follow below rules:
                - Action should be calling the 'ErrorHandler', who is handling errors.
                - AgentInputs will be the explanation about the error.
                If you successfully finish all tasks,
                - Action should be calling the agent 'NavigationHandlerMain' and AgentInput should be a ***very short message*** mentioning task is completed.
                """,
                ),
            ]
        )
    
    def setup_llm_chains(self):
        """Chain 1: LLM for output parser"""
        llm_for_outputparser = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=output_struct_NavigationalHandlerSecAgent)
        self.final_output_chain  = self.Navigational_supervisor_prompt | llm_for_outputparser | self.parser_output

    def process_user_request(self, request):
        self.given_task = request

        response_json = self.final_output_chain.invoke({"messages": self.chat_messages,
                                                        "given_task": self.given_task,
                                                        "agent_members": self.agent_members, 
                                                        "agent_info": self.agent_info, 
                                                        "special_info": self.special_info, 
                                                        "format_instructions": self.parser_output.get_format_instructions()})

        Thought_NHS = response_json["Thought"]
        Action_NHS = response_json["Action"]
        AgentInput_NHS = response_json["AgentInput"]

        formatted_msg = f"Thought: {Thought_NHS}, Action: {Action_NHS}, AgentInput: {AgentInput_NHS}"
        self.chat_messages.append(AIMessage(content=formatted_msg))

        return response_json

    def add_observations(self, observation):
        observation = observation
        self.chat_messages.append(HumanMessage(content=observation))

    def reset_chat_messages(self):
        self.chat_messages = []

# if __name__ == "__main__":
#     dd_agent = NavigationalHandlerSec()
#     dd_agent.process_user_request()
