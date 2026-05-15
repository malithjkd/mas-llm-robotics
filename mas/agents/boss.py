"""Boss"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class output_struct_boss(BaseModel):
    Thought: str = Field(description="you should always think step by step to determine what to do next according to the previous observation and the given task")
    Action: str = Field(description="The name of the next agent to activate")
    AgentInput: str = Field(description="This should be the input arguments to the agent separated by commas")

class BossAgent():
    """
    This agent is the boss who responsible for managing the conversation between the members under it self
    to fullfill a navigational task. And also he is the one who filterout user requests.
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
        # chat messages and the current ai response
        self.chat_messages = []
        self.current_ai_response = None

    def load_additional_info(self):
        self.base_path = os.path.dirname(__file__)

        with open(os.path.join(self.base_path, "Boss_supportive_info.txt"), 'r') as file:
            self.supportive_info = file.read()

        with open(os.path.join(self.base_path, "Agent_description_Boss.txt"), 'r') as file:
            self.member_info = file.read()
 
        self.members_agent_lst = ["Finisher", "Speaker", "TopPlanner"]

    def load_prompts(self):
        # Set up the agent prompt
        # Set up the agent prompt response
        self.Boss_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a robot boss who is responsible on managing the conversation between the agents and also filterout the user requests.
                - Robot current position is: {robot_current_position}.\n
                - The agents works under you: {members}.\n
                - The details about the agents: {members_info}\n
                - The rules and supportive info: {rules_and_supportive_info}\n.
                - The user request is: {user_request}.\n
                Please follow these rules when you give a response:
                - You are given the request from the user.
                - First decide whether the request is a ***Question or a Navigational task.***
                - Then follow the given rules carefully before you give your response.
                - Choose the most relevant agent to call next using the conversation history. Provide the input arguments to those agents.
                - Strictly follow the following format when you give your response: {format_instructions}
                    ... (this Thought/Action/AgentInputs can repeat N times)
                - After each agent call, you will be provided the result of each agent with a explanation as the Obseravtion.\n
                """),
                MessagesPlaceholder(variable_name="messages"),
                (
                "system",
                """
                Given the conversation above, you should decide the next agent to call according to the given rules.
                Follow the rules strickly.
                You should extract necessary details from the above conversation when you provide the input arguments to the agent you are going to call next.
                Follow the correct format given to you.
                """,
                ),
            ]
        )
    
    def setup_llm_chains(self):
        """Chain 1: LLM for the response"""
        llm_for_response = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=output_struct_boss)
        self.final_output_chain  = self.Boss_prompt | llm_for_response | self.parser_output

    def process_user_request(self, request, robot_current_position="None"):
        user_request = request

        response_json = self.final_output_chain.invoke({ "messages": self.chat_messages, 
                                                        "robot_current_position": robot_current_position,
                                                        "members": ", ".join(self.members_agent_lst),
                                                        "members_info": self.member_info,
                                                        "rules_and_supportive_info": self.supportive_info, 
                                                        "user_request": user_request,
                                                        "format_instructions": self.parser_output.get_format_instructions()})
            
        Thought_BOSS = response_json["Thought"]
        Action_BOSS = response_json["Action"]
        AgentInput_BOSS = response_json["AgentInput"]

        formatted_msg = f"Thought: {Thought_BOSS}, Action: {Action_BOSS}, AgentInput: {AgentInput_BOSS}"
        self.chat_messages.append(AIMessage(content=formatted_msg))

        return response_json
    
    def add_observations(self, observation):
        observation = observation
        self.chat_messages.append(HumanMessage(content=observation))
    
    def reset_chat_messages(self):
        self.chat_messages = []

# if __name__ == "__main__":
#     dd_agent = BossAgent()
#     request = str(input("Enter the user request: "))
#     robot_position = str(input("Enter the robot current position: "))

#     while True:
#         ai_response = dd_agent.process_user_request(request, robot_current_position=robot_position)
#         print(f"AI: {ai_response}\n")
#         observation = str(input("Enter the observation: "))
#         dd_agent.add_observations("Observation: " + observation)
