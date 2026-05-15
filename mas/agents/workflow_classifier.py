"""Workflow Classifier Agent"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class output_struct_workflow_classifier(BaseModel):
    Thought: str = Field(description="you should always think step by step to determine what to do next according to the previous observation and the given task plan")
    Action: str = Field(description="The name of the next agent to activate")
    AgentInput: str = Field(description="Input arguments to the agent separated by commas")

class WorkflowClassifierAgent():
    """
    This agent is the workflow classifier who responsible for managing the conversation between the members under it self
    to fullfill a navigational plan.
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

        with open(os.path.join(self.base_path, "Workflowclassifier_supportive_info.txt"), 'r') as file:
            self.supportive_info = file.read()

        with open(os.path.join(self.base_path, "Agent_description_Workflowclassifier.txt"), 'r') as file:
            self.member_info = file.read()
 
        self.members_agent_lst = ["NavigationsupervisorMain", "ObjectSearcher", "ObjectGrabber", "QuestionAnswerer", "Speaker", "Finisher"]

    def load_prompts(self):
        # Set up the agent prompt
        # Set up the agent prompt response
        self.Boss_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a robot supervisor who is responsible on managing the conversation between the agents to fullfill a task plan.
                - User request: {user_request}.\n
                - The given task plan is: {task_plan}.\n
                - The agents works under you: {members}.\n
                - The details about the agents: {members_info}\n
                - The rules and supportive info: {rules_and_supportive_info}\n.
                Please follow these rules when you give a response:
                - You are given the user request and also task plan proposed. You should follow that task plan step by step to fullfill the user request.
                - Follow the given rules carefully before you give your response.
                - Carefully go throught the rules and supportive info given to you and choose the most relevant agent to call next. Also use the conversation history for the decision.
                - Make sure provide correct input arguments to the agent you are going to call next.
                - Strictly follow the following format when you give your response: {format_instructions}\n
                    ... (this Thought/Action/AgentInputs can repeat N times)
                - After each agent call, you will be provided the result of each agent with a explanation as the Obseravtion.\n
                """),
                MessagesPlaceholder(variable_name="messages"),
                (
                "system",
                """
                Given the conversation above and the task plan, you should decide the next agent to call according to the given rules.
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
        self.parser_output = JsonOutputParser(pydantic_object=output_struct_workflow_classifier)
        self.final_output_chain  = self.Boss_prompt | llm_for_response | self.parser_output

    def process_user_request(self, user_request, task_plan):
        user_request = user_request
        task_plan = task_plan

        response_json = self.final_output_chain.invoke({ "messages": self.chat_messages,
                                                        "user_request": user_request,
                                                        "task_plan": task_plan,
                                                        "members": ", ".join(self.members_agent_lst),
                                                        "members_info": self.member_info,
                                                        "rules_and_supportive_info": self.supportive_info,
                                                        "format_instructions": self.parser_output.get_format_instructions()})
            
        Thought_WC = response_json["Thought"]
        Action_WC = response_json["Action"]
        AgentInput_WC = response_json["AgentInput"]

        formatted_msg = f"Thought: {Thought_WC}, Action: {Action_WC}, AgentInput: {AgentInput_WC}"
        self.chat_messages.append(AIMessage(content=formatted_msg))

        return response_json
    
    def add_observations(self, observation):
        observation = observation
        self.chat_messages.append(HumanMessage(content=observation))
    
    def reset_chat_messages(self):
        self.chat_messages = []

# if __name__ == "__main__":
#     dd_agent = WorkflowClassifierAgent()
#     user_request = "The robot current position is R1, user request is to take him to kuka robot, this is a navigational command."
#     task_plan = "1. I'm currently in R1. Navigate to the room where I can find the kuka robot.\n2. Search for the kuka.\n3. Inform the user about the final update (failure or success)."

#     while True:
#         ai_response = dd_agent.process_user_request(user_request=user_request, task_plan=task_plan)
#         print(f"AI: {ai_response}\n")
#         observation = str(input("Enter the observation: "))
#         dd_agent.add_observations("Observation: " + observation)
