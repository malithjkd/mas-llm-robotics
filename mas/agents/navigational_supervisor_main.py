"""Navigation_supervisor_main"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class output_struct_navigation_supervisor_main(BaseModel):
    final_response: str = Field(description="Response from the llm.")

class NavigationSupervisorMainAgent():
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
        # chat messages and the current ai response
        self.chat_messages = []
        self.current_ai_response = None

    def load_additional_info(self):
        self.base_path = os.path.dirname(__file__)

        with open(os.path.join(self.base_path, "Agent_description_navigation.txt"), 'r') as file:
            self.member_agents_info = file.read()
 
        self.members_agent_lst = ["FinalDestinatioIdentifier", "CurrentPositionIdentifier", "WaypointGenerator", "FinalNavigationalPlanner"]

    def load_prompts(self):
        # Set up the agent prompt
        self.Navigational_supervisormain_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a navigational supervisor who has the responsibility on navigating the robot.
                You are responsible for provide the correct call order of the members to manage the conversation between them.
                The members works under you are: {members}.\n
                The details of the members are: {supportive_info}\n
                Please follow these rules when you give a response:
                - According to the user command you have to find the robot current position, destination position, waypoints and from those information you have to plan the navigation.
                - You have to fetch the information from the members that provided to you.
                - Your responsibility is to plan the order of the members.
                - When you give the order of the members, strickly follow the details of the members since some members can depend on the output of the other members.
                - You should also provide the input arguments to that members.
                - Strictly follow the details and response with the following format to provide the members assigned order:
                    --Thought: <explanation how you decide the member order>
                    --1. first member: <name: name of the member> <inputSchema: 'input arguments to the member'>
                    --2. second member: <name: name of the member> <inputSchema: 'input arguments to the member'>
                    ...(and other members also should assigned)...
                - You should consider the following conversation from the user, inorder to provide the member assignment order with the input arguments.\n
                """),
                MessagesPlaceholder(variable_name="messages"),
                (
                "system",
                """
                Given the conversation above, you should decide the order of the members to call and the input arguments to provide to that each members.
                - when you get the details of the doors, Make sure you follow the entire conversation.
                When you give your response, make sure that you follow the correct format and also the correct order of the members with correct input arguments.
                """,
                ),
            ]
        )
    
    def setup_llm_chains(self):
        """Chain 1: LLM for the response"""
        llm_for_response = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=output_struct_navigation_supervisor_main)
        self.final_output_chain  = self.Navigational_supervisormain_prompt | llm_for_response

    def process_user_request(self, request):
        add_info = request
        self.chat_messages.append(HumanMessage(content=add_info))

        response = self.final_output_chain.invoke({"messages": self.chat_messages, 
                                                    "members": ", ".join(self.members_agent_lst),
                                                    "supportive_info": self.member_agents_info})
        
        self.current_ai_response = response.content

        return self.current_ai_response
    
    def reset_chat_messages(self):
        self.chat_messages = []

# if __name__ == "__main__":
#     dd_agent = NavigationSupervisorMainAgent()
#     info = "I need to go to a place where KUKA is placed."
#     ai_response = dd_agent.process_user_request(info)
#     print(f"User: {dd_agent.chat_messages}")
#     print(f"AI: {ai_response}\n")

#     info = "Door 2 is closed."
#     ai_response = dd_agent.process_user_request(info)
#     print(f"User: {dd_agent.chat_messages}")
#     print(f"AI: {ai_response}\n")

#     info = "Door 3 is closed."
#     ai_response = dd_agent.process_user_request(info)
#     print(f"User: {dd_agent.chat_messages}")
#     print(f"AI: {ai_response}\n")

#     info = "Door 4 is closed."
#     ai_response = dd_agent.process_user_request(info)
#     print(f"User: {dd_agent.chat_messages}")
#     print(f"AI: {ai_response}\n")

#     dd_agent.reset_chat_messages()
#     info = "I need to go to the study room"
#     ai_response = dd_agent.process_user_request(info)
#     print(f"User: {dd_agent.chat_messages}")
#     print(f"AI: {ai_response}\n")

# # Test with various inputs
# test_cases = [
#     "I need to go to the room 1", 
#     "I want to reach the room with the KUKA robot arm.",
#     "I am looking for a screw driver.",
#     "I need to find a person and he is in a study area.",
#     "I need to wash my hands.", 
#     "I need to play some computer games", 
#     "I need to go to the washroom. I wanna pee."
# ]
