"""RAG agent for final destination identifier"""

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
class RAGToolInput_DestinationDecisionAgent(BaseModel):
    query: str = Field(description="""  Query string to search for relevant information.
                                        If user need information about environment map which include how each rooms and cooridor is placed in the map, use the query as 'environment map'.
                                        If user need information about each room like equipments in the room or for which purpose each room is using, use the query as 'room description'.""")

def get_DestinationDecisionAgent_tools(retriever, content_chunks):
    @tool("RAGTool", args_schema=RAGToolInput_DestinationDecisionAgent)
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

class DestinationDecisionAgent_output_struct(BaseModel):
    Explanation: str = Field(description="The explaination of how user has decided the destination or what is the error raised. Make this very short.")
    Destination: str = Field(description="The final destination of the robot explained in the message. Add only the final destination. Should be one of 'R1, R2, R3, C, unknown'")

class DestinationDecisionAgent():
    """
    This agent is responsible for making decisions about the final destination of the robot.
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
        self.RAGTool = get_DestinationDecisionAgent_tools(self.retriever, self.content_chunks)
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
        self.destination_decide_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are an intelligent assistant helping a robot to decide its correct destination based on the user's request.
                Your main goal is to identify the final destination location where the robot should go.
                You are also provided with supportive information about the environment. If needed, you can use a tool called `RAGTool` to fetch more information from the environment description.
                Please follow these rules:
                - If you need more context or details to determine the destination, **call the relevant tool** which have provided to you.
                - Once you have enough information and determine the correct destination, **return** the output answer with a better explanantion how you decide the destination.
                - If you are not sure about the destination, provide that destination as 'unknown'.
                - But before providing 'unknown', you should try to use the tool to get more information.
                - Your final destination should be Room1(R1) or Room2(R2) or Room3(R3) or Corridor(C). You should find the correct destination from the user request and the supportive information.
                - Strictly follow the follwing format for your final answer:
                    -- Final destination: <destination. Should be one of R1, R2, R3, C, unknown>
                    -- Explanation: <explanation of how you decide the destination. This should be a short explanation.>
                - If the final destination cannot be identified correctly, add 'Error occurs, ' before the explnanation."""),
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
                    Your main goal is to format the output by the user in a structured way as a json object.
                    When you add the final destination, make sure you use following words:
                    -- Room1: 'R1', Room2: 'R2', Room3: 'R3', Corridor: 'C', Unknown: 'unknown'
                    The output should be in the following format: {format_instructions}
                    Strictly follow the format and do not add any extra information.
                """),
                ("human", "{output_by_the_agent}"),
            ]
        )
    
    def setup_llm_chains(self):
        # Set up the LLM and tools
        llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini", openai_api_key=self.openai_api_key)

        """Chain 1: LLM for decision making"""
        tools = [self.RAGTool]
        # Create the agent
        agent = create_openai_tools_agent(llm, tools, self.destination_decide_prompt)
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
        )

        """Chain 2: LLM for output parser"""
        self.parser_output = JsonOutputParser(pydantic_object=DestinationDecisionAgent_output_struct)
        self.final_output_chain  = self.output_struct_prompt | llm | self.parser_output

    def process_user_request(self, user_request: str):
        """
        Process the user request and return the final destination and explanation.
        """
        # Get the output from the agent executor
        agent_response_str = self.agent_executor.invoke({"user_request": user_request})
        # Get the output from the final output chain
        final_output_json = self.final_output_chain.invoke({"output_by_the_agent": agent_response_str['output'], 
                                                       "format_instructions": self.parser_output.get_format_instructions()})

        return final_output_json
   
DDagent = DestinationDecisionAgent()

# Test with various inputs
test_cases = [
    # "I need to go to the room 1", 
    "I want to reach the room with the KUKA robot arm.",
    # "I am looking for a screw driver.",
    # "I need to find a person and he is in a study area.",
    # "I need to wash my hands.", 
    # "I need to play some computer games", 
    # "I need to go to the washroom. I wanna pee."
]

for test_case in test_cases:
    response = DDagent.process_user_request(test_case)
    print(f"User Input: {test_case}")
    print(f"Response: {response}\n")