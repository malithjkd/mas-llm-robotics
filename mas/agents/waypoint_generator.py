"""RAG agent for Waypoint Generator agent"""

import os
from dotenv import load_dotenv

# ---------fetching necessary documents-----------
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool

# Define RAGtool tool
class RAGToolInput_WaypointGenerationAgent(BaseModel):
    query: str = Field(description="""  Query string to search for relevant information. 
                                        If user need information about environment map which include how each rooms and cooridor is placed in the map, use the query as 'environment map'.
                                        If user need information about each room like equipments in the room or for which purpose each room is using, use the query as 'room description'. """)

def get_WaypointGenerationAgent_tools(retriever, content_chunks):
    @tool("RAGTool", args_schema=RAGToolInput_WaypointGenerationAgent)
    def RAGTool(query: str) -> str:
        """
        RAG (Retrieval-Augmented Generation) tool that retrieves relevant info based on the query.
        Output: Relevant content retrieved from the resources.
        """
        # Use your retriever here
        retrieved_docs = retriever.get_relevant_documents(query)
        title_no = retrieved_docs[0].metadata['section_number']
        relevant_content = content_chunks[title_no-1]

        return relevant_content
    
    return RAGTool

class WaypointGenerationAgent_output_struct(BaseModel):
    WaypointSequence: str = Field(description="""The sequence of waypoints that the robot should follow to reach the final destination.
                                                 The waypoints should be in this format: R1->R2->R3->C""")
    doorSequence: str = Field(description=""" The doors that the robot going to go through.
                                              The doors sequence should gives in the correct order according to the path.
                                              The doors should be in this format: D1->D2->D3""")
    FinalSequence: str = Field(description="""Concatanate Door sequence with the waypoint sequence.
                                              The order should be like this: 1st waypoint->1st door->2nd waypoint->second door.........""")

class WaypointGenerationAgent():
    """This agent is responsible for generate the relevant waypoints to reach the final destination. 
       It uses a RAG tool to retrieve relevant information based on user queries.
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

        # load the database
        self.load_database()
        # load the RAG tool
        self.RAGTool = get_WaypointGenerationAgent_tools(self.retriever, self.content_chunks)
        # load the prompts
        self.load_prompts()
        # Set up the LLM
        self.setup_llm_chains()

    def load_database(self):
        self.base_path = os.path.dirname(__file__)

        documents = []
        self.content_chunks = []
        # Add content from local file
        with open(os.path.join(self.base_path, "Environment_description.txt"), 'r') as file:
            local_content = file.read()

            # Split manually if needed
            section_chunks = local_content.split("//new section//")  # delimiter for split chunks by sections
            # split the content and the titles and put them in to the list
            section_no = 1
            for section in section_chunks:
                whole_chunk = section.split("//content//")
                title_chunk = whole_chunk[0].strip()
                content_chunk = whole_chunk[1].strip()
            
                self.content_chunks.append(content_chunk)
                documents.append(Document(page_content=title_chunk, metadata={"section_number": section_no}))
                section_no += 1

        # Create the vector store directly from these chunks
        embedding = OpenAIEmbeddings()
        vectorStore = FAISS.from_documents(documents, embedding=embedding)
        self.retriever = vectorStore.as_retriever(search_kwargs={'k': 1})

    def load_prompts(self):
        # Set up the agent prompt
        self.waypoint_decide_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are an intelligent assistant helping a robot to decide the correct path where the robot should go through in order to reach the destination.
                If needed, you can use a tool called 'RAGTool' to fetch more supportive information from the environment map.
                Please follow these rules:
                - User will provide the current location of the robot and the final destination.
                - You should **call the relevant tool** in order to get the explanation of the map and the example scenarios. Use that map and examples carefully to decide the path.
                - Once you have enough information and determine the correct path, **return** the output answer with a better explanantion how you decide the path.
                - Make sure you consider the information about the doors which are give by the user. That means you should consider whether the doors are open or closed in order to generate the path.
                - If one path is blocked due to some doors are closed, you should find another path.
                - Your final path should be like 'R1->R2->R3->C' and Doors sequence should be like 'D2->D3->D4'. 
                - You should find the correct path from the user request and the supportive information.
                - If user doesn't mention about the doors, you should assume that all the doors are open.
                - Strictly follow the follwing format for your final answer:
                    -- Thought: <explanation of how you decide the path>
                    -- Final path: <path you generate> This should include all the rooms and corridors in the path starting from the starting position to the final destination.
                    -- Doors sequence: <Sequence of the doors that the robot going to go through according to the path. If there are no doors in the path, you should provide 'None'>
                - If ***all the available paths are blocked*** due to some doors in the path are closed, provide the output as 'Error occurs, unknown'.
                """),
                ("human", "{user_request}"),
                ("ai", "{agent_scratchpad}"),
            ]
        )

        # Set up the output struct prompt
        self.output_struct_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """ You are an intelligent assistant, helping a robot to structure the output in a specific format.
                    When you add the path (waypoints) given by the user, make sure you use following words:
                    -- Room1: 'R1', Room2: 'R2', Room3: 'R3', Corridor: 'C'
                    -- Unknown: 'Error occured, cannot proceed to the destination since all available paths are blocked'.
                    -- Door1: 'D1', Door2: 'D2', Door3: 'D3', Door4: 'D4'
                    The output should be in the following format: {format_instructions}
                    Strictly follow the format and do not add any extra information.
                """),
                ("human", "{output_by_the_agent}"),
            ]
        )
    
    def setup_llm_chains(self):
        # Set up the LLM and tools
        llm_for_waypointgeneration = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=self.openai_api_key)

        """Chain 1: LLM for generate waypoints"""
        tools = [self.RAGTool]
        # Create the agent
        agent = create_openai_tools_agent(llm_for_waypointgeneration, tools, self.waypoint_decide_prompt)
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
        )

        """Chain 2: LLM for output parser"""
        llm_for_outputparser = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=WaypointGenerationAgent_output_struct)
        self.final_output_chain  = self.output_struct_prompt | llm_for_outputparser | self.parser_output

    def process_user_request(self, user_request: str):
        """
        Process the user request and return the final destination and explanation.
        """
        # Get the output from the agent executor
        agent_response_str = self.agent_executor.invoke({"user_request": user_request})
        # print(f"agent_first_response_str : \n{agent_response_str}\n\n")
        # Get the output from the final output chain
        final_output_json = self.final_output_chain.invoke({"output_by_the_agent": agent_response_str['output'], 
                                                       "format_instructions": self.parser_output.get_format_instructions()})
        
        # print(f"agent_second_response_str : \n{final_output_json}\n")

        return final_output_json
    
# WGagent = WaypointGenerationAgent()

# # Test with various inputs
# test_cases = [
#     "Robot current position is R3, Robot destination is R1, and the D3 and D4 are closed.",
#     # "I need to go to the room 2 and im currently in the room 1", 
#     # "I need to go to the room 3 and im currently in the room 1",
#     # "I need to go to the room 3 and im currently in the room 1 and the door 2 is closed",
#     # "I need to go to the room 3 and im currently in the corridor and the door 4 is closed",
#     # "I need to go to the room 2 and im currently in the room 1 and the door 2 is closed",
#     # "I need to go to the room 1 and im currently in room 1",
#     # "I need to go to the room 2 and im currently in room 2",
#     # "I need to go to the room 3 and im currently in room 3",
#     # "I need to go to the C and im currently in C",
# ]

# for test_case in test_cases:
#     response = WGagent.process_user_request(test_case)
#     print(f"User Input: {test_case}")
#     print(f"Response: {response}\n")
#     print("\n", "-" * 50, "\n")