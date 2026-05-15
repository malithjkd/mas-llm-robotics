"""Door Status Checker Agent"""

import os
import base64
from dotenv import load_dotenv
from typing import Optional
import cv2
import numpy as np

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

class output_struct_DoorStatusCheckerAgent(BaseModel):
    Door_label: str = Field(description="The label of the door (e.g., D1, D2, etc.)")
    Door_status: str = Field(description="The status of the door (open or closed)")
    Explanation: str = Field(description="""Explanation should follow below rules:
                                            - If door is 'closed': 'Error Occured. Door <door_label> is closed. Cannot proceed further'.
                                            - If door is 'open': 'Door <door_label> is open'.
                                            """)

class DoorStatusCheckerAgent:
    """
    Agent that analyzes door images to identify door label and status using GPT-4 Vision.
    """
    def __init__(self):
        # load the api keys
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # load the prompt
        self.load_prompts()
        # setup the chain
        self.setup_llm_chains()

    def load_prompts(self):
        """Set up the image processing chain with prompt template."""
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an intelligent door status checker agent.
                Your task is to analyze the door image and determine the door label and its status.
                """),  
            ("human", [
                {"type": "text", "text": 
                    """ Analyze this door image carefully and Identify:
                        - The nearest door number which visible in the image near to the door (1-4). Here 1 means D1 likewise.
                        - Whether the door is open or closed
                    
                        Respond strictly in the following format:
                        {format_instructions}.\n
                        
                        Important Notes:
                        - Only use door labels D1, D2, D3, or D4
                        - Status must be either 'open' or 'closed'
                        - Check the image carefully to determine the details"""
                },
                {"type": "image_url", "image_url": {"url": "{image_url}"}}
            ])
        ])

    def setup_llm_chains(self):   
        llm_parser = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=200, openai_api_key=self.openai_api_key)
        self.json_parser = JsonOutputParser(pydantic_object=output_struct_DoorStatusCheckerAgent) 
        self.analysis_chain = (self.prompt_template | llm_parser | self.json_parser)

    def process_user_request(self, image: np.ndarray) -> Optional[dict]:
        # Encode image to base64
        base64_image = self._encode_image(image)

        image_url = f"data:image/jpeg;base64,{base64_image}"
        response_json = self.analysis_chain.invoke({
                                                    "image_url": image_url,
                                                    "format_instructions": self.json_parser.get_format_instructions()
                                                    })
        
        return response_json

    def _encode_image(self, image: np.ndarray) -> Optional[str]:
        # Encode image as JPEG in memory
        success, buffer = cv2.imencode('.jpg', image)
        # Convert buffer to base64 string
        return base64.b64encode(buffer).decode("utf-8")


# if __name__ == "__main__":
#     agent = DoorStatusCheckerAgent()
    
#     test_image_path = r"<test-image-path>"
#     image = cv2.imread(test_image_path)

#     analysis_result = agent.process_user_request(image)
#     print("Json Result: ")
#     print(analysis_result)