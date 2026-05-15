"""Error handler"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class output_struct_ErrorHandlerAgent(BaseModel):
    Thought: str = Field(description="you should always think to determine what to do next according to the given details.")
    Action: str = Field(description="The name of the next agent to activate")
    AgentInput: str = Field(description="This should be the input arguments to the agent separated by commas")

class ErrorHandlerAgent():
    """
    This agent is responsible for handling the errors.
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
        # info about error handling
        with open(os.path.join(self.base_path, "Error_handling_info.txt"), 'r') as file:
            self.error_handling_doc = file.read()

    def load_prompts(self):
        # Set up the agent prompt
        self.Error_handler_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a intelligent Error handler for a robot.
                You are responsible for carefully refer the error and assign the agent for deal with the error.
                The error is: {error_occured}.\n
                The error handling information: {error_handling_doc}.\n
                Please strickly follow below rules when you give a response:
                - You should always think to determine what to do next according to the given details.
                - You should decide the type of error occured according the error handling info and assign the correct agent to handle the error.
                - When you assign the agent, you should provide the correct input arguments to the agent.
                - When you give the response, you should strictly follow the format below:
                    -- {format_instructions}.\n
                """),   
            ]
        )
    
    def setup_llm_chains(self):
        """Chain 1: LLM for output parser"""
        llm_for_outputparser = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=output_struct_ErrorHandlerAgent)
        self.final_output_chain  = self.Error_handler_prompt | llm_for_outputparser | self.parser_output

    def process_user_request(self, request):
        self.error_occured = request

        response_json = self.final_output_chain.invoke({"error_occured": self.error_occured,
                                                        "error_handling_doc": self.error_handling_doc,
                                                        "format_instructions": self.parser_output.get_format_instructions()})

        return response_json

    def add_observations(self, observation):
        observation = observation
        self.chat_messages.append(HumanMessage(content=observation))

    def reset_chat_messages(self):
        self.chat_messages = []

# if __name__ == "__main__":
#     dd_agent = ErrorHandlerAgent()
#     info = "Error Occured. Door 2 is closed. Cannot proceed further."
#     ai_response = dd_agent.process_user_request(info)
#     print(f"User: {info}")
#     print(f"AI: {ai_response}\n")

#     info = "Error occured. All the paths are blocked."
#     ai_response = dd_agent.process_user_request(info)
#     print(f"User: {info}")
#     print(f"AI: {ai_response}\n")

#     info = "Error occured. Cannot find the destination."
#     ai_response = dd_agent.process_user_request(info)
#     print(f"User: {info}")
#     print(f"AI: {ai_response}\n")