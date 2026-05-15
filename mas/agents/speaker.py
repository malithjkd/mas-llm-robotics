"""Speaker Agent"""

import os
from dotenv import load_dotenv

"""LLM"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage

"""ELEVEN LABS"""
from elevenlabs.client import ElevenLabs
from elevenlabs import play, save

class output_struct_speaker(BaseModel):
    Thought: str = Field(description="you should always think step by step to give the final output")
    Action: str = Field(description="The reply which speaker should give to the user")

class SpeakerAgent():
    """
    This agent is the speaker which is a user friendly speaking agent.
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
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        # load the parameter base
        self.load_additional_info()
        # load the prompts
        self.load_prompts()
        # Set up the LLM
        self.setup_llm_chains()
        # chat messages and the current ai response
        self.chat_messages = []

    def load_additional_info(self):
        self.base_path = os.path.dirname(__file__)

        female_voice_ids = {
                                "Bella": "EXAVITQu4vr4xnSDxMaL",
                                "Rachel": "21m00Tcm4TlvDq8ikWAM",
                                "Sarah": "EXAVITQu4vr4xnSDxMaL",
                                "Domi": "AZnzlk1XvdvUeBnXmlld",
                                "Alice": "Xb7hH8MSUJpSbSDYk0k2",
                                "Dorothy": "ThT5KcBeYPX3keUQqHPh",
                                "Emily": "LcfcDJNUP1GQjkzn1xUU",
                                "Matilda": "XrExE9yKIg1WjnnlVkGX",
                                "Lily": "pFZP5JQG7iQjIQuC4Bku",
                                "Nicole": "piTKgcLEGmPE4e6mEKli",
                                "Serena": "pMsXgVXv3BLzUgSXRplE"
                                }
        
        self.selected_voice_id = female_voice_ids["Bella"]  # Bella is better in voice
        with open(os.path.join(self.base_path, "Speaker_supportive_info.txt"), 'r') as file:
            self.supportive_info = file.read()

    def load_prompts(self):
        # Set up the agent prompt
        # Set up the agent prompt response
        self.Speaker_prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a user ***friendly robot*** who is responsible for speak with user.
                Your task is to say the message to the user, provided by operating system of the robot,in a very friendly manner.
                - The message: {message_to_user}.\n
                - Supportive info: {supportive_info}.\n
                You should analyse the message and you should inform that message to the user in a friendly manner.
                Change your response accordingly such that it is very friendly.
                When you give your response, follow the supportive info provided to you.
                - Don't put words like 'oh no', 'hurrey'. But apologising is accepted.
                - Strictly follow the following format when you give your response: {format_instructions}\n
                """),
            ]
        )
    
    def setup_llm_chains(self):
        """Chain 1: LLM for the response"""
        llm_for_response = ChatOpenAI(temperature=0.3, model="gpt-4o-mini", openai_api_key=self.openai_api_key)
        self.parser_output = JsonOutputParser(pydantic_object=output_struct_speaker)
        self.final_output_chain  = self.Speaker_prompt | llm_for_response | self.parser_output

        self.speaker_client = ElevenLabs(api_key=self.elevenlabs_api_key)

    def process_user_request(self, user_request):
        message_to_user_from_agent = user_request

        response_json = self.final_output_chain.invoke({ "message_to_user": message_to_user_from_agent,
                                                        "supportive_info": self.supportive_info,
                                                        "format_instructions": self.parser_output.get_format_instructions()})

        return response_json

    def speak(self, message):

        try:
            audio = self.speaker_client.text_to_speech.convert(
                text=message,
                voice_id=self.selected_voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )

            play(audio)

            output_filename = os.path.join(self.base_path, "speaker_output.mp3")

            save(audio, output_filename)

        except Exception as e:
            print(f"An error occurred: {e}")


# if __name__ == "__main__":
#     dd_agent = SpeakerAgent()
#     user_request = [
#                     "Hammer is successfully grabbed from the room 2",
#                     "Error, cannot proceed further due to all path blocked",
#                     "The final destination is reached, the task is completed successfully",
#                     "Cannot find the hammer in the room 2",
#                     "This task is out of the robot capabilities.", 
#                     "I cannot go further due to the door closure",
#                     "Error, cannot proceed further due to all path blocked",
#                     ]

#     for request in user_request:
#         ai_response_json = dd_agent.process_user_request(user_request=request)
#         ai_response = ai_response_json["Action"]

#         print(f"User Request: {request}")
#         print(f"AI: {ai_response}\n")

#         dd_agent.speak(ai_response)

#         print("" + "="*50 + "\n")

