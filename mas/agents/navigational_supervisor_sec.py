"""Navigational planner secondary agent"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class output_struct_navigation_supervisor_sec(BaseModel):
    Thought: str = Field(description="Thought given in the user input")
    Action: str = Field(description="The next agent to activate given in the user input")
    AgentInput: str = Field(description="Agent inputs given in the user input")

class NavigationSupervisorSecAgent():
    """
    This agent is responsible for managing the conversation between the members under it self
    to fullfill a navigational task.
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
        
        self.agent_members_lst = ["FinalDestinatioIdentifier", "CurrentPositionIdentifier", "WaypointGenerator", "FinalNavigationalPlanner"]

        self.example_task_plan = """ 
                                --Thought: To navigate the robot to the destination where KUKA is placed, we need to first identify the final destination. Then, we need to determine the robot's current position. With both the current position and the destination known, we can generate the waypoints. Finally, we will plan the final navigation considering the closed door.

                                --1. first member: <name: FinalDestinatioIdentifier> <inputSchema: 'Robot needs to go to a place which has KUKA.'>
                                --2. second member: <name: CurrentPositionIdentifier> <inputSchema: 'None'>
                                --3. third member: <name: WaypointGenerator> <inputSchema: 'The robot is currently in < put the current position here>, the destination is where KUKA is placed, and Door 2 is closed.'>
                                --4. fourth member: <name: FinalNavigationalPlanner> <inputSchema: 'Robot currently in <current position>, Robot need to go to where KUKA is placed, Door 2 is closed. The path you have to follow is <waypoints>. The door sequence is <door sequence>.'>
                                """

    def load_prompts(self):
        # Set up the agent prompt response
        self.Navigational_supervisorsec_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a navigational supervisor who is responsible on executing a given task plan to manage the conversation between the agents to navigate a robot.
                The agents works under you: {members}.\n
                The agentic task plan: {task_plan}\n.
                Please follow these rules when you give a response:
                - You are given a order of the agents to call and the input arguments to provide to that each agent. 
                - You should continue the conversation with the agents in the order given above and provide the input arguments to those agents using the conversation history.
                - when you calling the agents, strickly follow the given order since some agents can depend on the output of the other agents.
                - Make sure you give the correct input argument to the agents.
                - Strictly follow the following format when you give your response:\n
                    Thought: <you should always think step by step to determine what to do next according to the given task plan>
                    Action: <The name of the next agent to activate>
                    AgentInputs: <Input arguments to the agent separated by commas>
                    ... (this Thought/Action/AgentInputs can repeat N times)
                - After each agent call, you will be provided the result of each agent with a explanation as the Obseravtion.\n
                """),
                MessagesPlaceholder(variable_name="messages"),
                (
                "system",
                """
                Given the conversation above, you should decide the next agent to call according to the given order.
                You should also extract some details from the above conversation when you provide the input arguments to the agent you are going to call next.
                - ex: You should give the current position of the robot when input specify the current position of the robot.
                Follow the correct format given to you.
                - If you get a error message from any of the agent, you should called the Action as 'ErrorHandler' and the 'AgentInputs' will be the explanation about the error.
                """,
                ),
            ]
        )

        # setup the output structure prompt
        self.Structured_output_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a supervisor who has the responsibility on structuring the user given input in a structured manner.
                The user given input is : {user_input}\n.
                Strictly follow the following format when you give your response:\n
                Output structure: {format_structure}\n.
                - Make sure you strickly follow the above format correctly.
                """), 
            ]
        )
    
    def setup_llm_chains(self):
        """Chain 1: LLM for response"""
        llm_for_response = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.final_response_chain  = self.Navigational_supervisorsec_prompt | llm_for_response

        """Chain 2: LLM for structured response"""
        llm_for_structured_output = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=output_struct_navigation_supervisor_sec)
        self.final_output_chain  = self.Structured_output_prompt | llm_for_structured_output | self.parser_output

    def process_user_request(self, request):
        task_plan = request

        response = self.final_response_chain.invoke({"messages": self.chat_messages,
                                                     "members": ", ".join(self.agent_members_lst),
                                                     "task_plan": task_plan})
        
        self.current_ai_response = response.content
        self.chat_messages.append(AIMessage(content=self.current_ai_response))

        formatted_json_response = self.final_output_chain.invoke({"user_input": self.current_ai_response,
                                                                  "format_structure": self.parser_output.get_format_instructions()})

        return formatted_json_response

    def add_observations(self, observation):
        observation = observation
        self.chat_messages.append(HumanMessage(content=observation))

    def reset_chat_messages(self):
        self.chat_messages = []

# if __name__ == "__main__":
#     dd_agent = NavigationSupervisorSecAgent()

#     while True:
#         info = dd_agent.example_task_plan
#         ai_response = dd_agent.process_user_request(info)
#         print(f"AI: {ai_response}")
#         observation = input("Observation: ")
#         dd_agent.add_observations(observation=observation)
