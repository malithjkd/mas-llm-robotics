"""Question Answering Agent"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool

# Define tool for Final Destination Identification
class ToolInput_FinalDestination(BaseModel):
    query: str = Field(description="Query string to identify the final destination.")

class ToolOutput_FinalDestination(BaseModel):
    FinalDestination: str = Field(description="The final destination of the robot, Strictly any of these 'Room1', 'Room2', 'Room3', 'Corridor', or 'Destination Cannot identify clearly'.")

def get_FinalDestination_tool():
    
    # Set up LLM and parser for structured output
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    parser = JsonOutputParser(pydantic_object=ToolOutput_FinalDestination)
    
    room_descriptions = """
                        2. Room Descriptions
                        Describe about each room and for which kind of purposes each room is using and what objects each room includes.

                        Room1: This is the study room for students. This includes the following objects in the room: sink, some tables and chairs to study, wheelchair robot.
                        Room2: This is the tool room. You can find various kinds of tools like hammers, soldering iron, solders, circuits, power supplies and various kinds of tools. If you need to find any physical tool, you need to come here.
                        Room3: This is the PLC room. This has 4 tables with PLC setups. And more importantly, the KUKA robot hand is in this room.
                        Corridor: Corridor has the shoerack.

                        The room Map is below:
                            o---------o---------o----o----o
                            |         |         |         |
                            |   R1    D2   R2   D3   R3   |
                            |         |         |         |
                            o D1 o----o---------o D4 o----|
                            |              C              |
                            o-----------------------------o

                        """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are tasked with identifying the answer from the given room descriptions based on the query."
         f"Based on the user query and the room information, determine which room contains the requested object or serves the requested purpose."
         f"Make sure to give a short answer."
         f"If you cannot exactly identify what is the answer, then output should be 'Destination Cannot identify clearly'"
         "\nRoom Information:\n{room_descriptions}\n"
         "Format your response according to: {format_instructions}"),
        ("human", "Query: {query}")
    ])
    
    chain = prompt | llm | parser
    
    @tool("FinalDestinationTool", args_schema=ToolInput_FinalDestination)
    def FinalDestinationTool(query: str) -> str:
        """
        Tool to identify the final destination based on the query.
        """
        result = chain.invoke({
            "query": query,
            "room_descriptions": room_descriptions,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result["FinalDestination"]
    
    return FinalDestinationTool

# Define tool for Waypoint Generation
class ToolInput_Waypoint(BaseModel):
    query: str = Field(description="Query string to generate waypoints.")

class ToolOutput_Waypoint(BaseModel):
    Waypoint: str = Field(description="The generated waypoint path based on the query or error message of cannot proceed.")

def get_Waypoint_tool():
    
    # Set up LLM and parser for structured output
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    parser = JsonOutputParser(pydantic_object=ToolOutput_Waypoint)
    
    environment_map = """
                        Environment includes 3 rooms (R1, R2, R3) and a corridor (C).
                        Doors are given as D1, D2, D3, D4.
                        Map is showing below:

                            o---------o---------o----o----o
                            |         |         |         |
                            |   R1    D2   R2   D3   R3   |
                            |         |         |         |
                            o D1 o----o---------o D4 o----|
                            |              C              |
                            o-----------------------------o

                        Robot can go a full round through the map like R1->R2->R3->C->R1.
                        The only doors which have access to the corridor are D1 and D4. Example paths:
                        Ex: Robot need to go from R1 to R3 => It can go through R1->D2->R2->D3->R3 or else R1->D1->C->D4->R3.
                        Ex: Robot need to go from R1 to R2 but door2 is closed => It can go through R1->D1->C->D4->R3->D3->R2 since D3 is still open.
                        """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         f"You are tasked with generating the path based on the environment map and navigation rules. "
         f"Based on the user query and the environment map, generate a the waypoint path to reach the destination. ex: 'First you have to go to the room1. Then through the room 2 you can go to room 3'"
         f"The waypoints should include rooms (R1, R2, R3, C) and doors (D1, D2, D3, D4) as needed. "
         f"Consider any door closure constraints mentioned in the query. If all the paths are blocked due to the door closure you response should be 'Cannot go to the <mentioned destination> due to all the doors are closed'"
         "\nEnvironment Information:\n{environment_map}\n"
         "Format your response according to: {format_instructions}"),
        ("human", "Query: {query}")
    ])
    
    chain = prompt | llm | parser
    
    @tool("WaypointTool", args_schema=ToolInput_Waypoint)
    def WaypointTool(query: str) -> str:
        """
        Tool to generate waypoints based on the query.
        """
        result = chain.invoke({
            "query": query,
            "environment_map": environment_map,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result["Waypoint"]
    
    return WaypointTool

class QuestionAnsweringAgent_output_struct(BaseModel):
    Explanation: str = Field(description="Explanation of how the question was answered.")
    FinalAnswer: str = Field(description="The final answer to the user's question. ")

class QuestionAnsweringAgent():
    """
    This agent is responsible for answering user questions related to navigation.
    It uses tools for final destination identification and waypoint generation.
    """
    def __init__(self):
        # Load the API keys
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Load tools
        self.FinalDestinationTool = get_FinalDestination_tool()
        self.WaypointTool = get_Waypoint_tool()
        # Load prompts
        self.load_prompts()
        # Set up the LLM
        self.setup_llm_chains()

    def load_prompts(self):
        # Set up the agent prompt
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are an intelligent assistant answering user questions related to navigation.
                You have access to tools for identifying the final destination and generating waypoints.
                
                Follow these rules for tool usage:
                - If the user asks "where is [something]" or "what is the location of [something]", use ONLY FinalDestinationTool
                - If the user asks "how to go to [somewhere]" or "generate path to [somewhere]", use BOTH FinalDestinationTool first to identify the destination user is mentioning, then WaypointTool to generate the path
                - Always provide a clear explanation of how you arrived at the answer
                
                format your final answer as:
                    -- Explanation: <how you determined the answer>
                    -- Final Answer to user: <The final answer>"""),
                ("human", "{user_request}"),
                ("ai", "{agent_scratchpad}"),
            ]
        )

        # Set up the output struct prompt
        self.output_struct_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are an assistant structuring the output in a specific format.
                Consider the user request and give the response according to the following format including the details provided to you
                The output should be in the following format: {format_instructions}
                Strictly follow the format and do not add any extra information."""),
                ("human", "{output_by_the_agent}"),
            ]
        )

    def setup_llm_chains(self):
        # Set up the LLM and tools
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=self.openai_api_key)

        # Chain 1: LLM for answering questions
        tools = [self.FinalDestinationTool, self.WaypointTool]
        agent = create_openai_tools_agent(llm, tools, self.qa_prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
        )

        # Chain 2: LLM for output parser
        self.parser_output = JsonOutputParser(pydantic_object=QuestionAnsweringAgent_output_struct)
        self.final_output_chain = self.output_struct_prompt | llm | self.parser_output

    def process_user_request(self, user_request: str):
        """
        Process the user request and return the final answer and explanation.
        """
        # print(f"[DEBUG] Received user request: {user_request}")

        # Use the agent executor to intelligently decide which tools to use
        
        agent_response = self.agent_executor.invoke({"user_request": user_request})
        agent_output = agent_response.get("output", "")
        # print(f"[DEBUG] Agent response: {agent_output}")

        # Parse the agent output using the structured output chain
        final_output_json = self.final_output_chain.invoke({
            "output_by_the_agent": agent_output,
            "format_instructions": self.parser_output.get_format_instructions()
        })
        
        # print(f"[DEBUG] Final structured output: {final_output_json}")
        return final_output_json


# # Example usage
# if __name__ == "__main__":
#     qa_agent = QuestionAnsweringAgent()

#     test_cases = [
#         "How do I get to place where kukka is there [if Door 2 is closed]?. User current position: R1",
#         "where is the kuka robot arm?",
#         "where should i go if i need to soldering",
#         "I need to go to the KUKA robot arm. User current position: R1",
#     ]

#     for test_case in test_cases:
#         response = qa_agent.process_user_request(test_case)
#         print(f"User Input: {test_case}")
#         print(f"Response: {response}\n")