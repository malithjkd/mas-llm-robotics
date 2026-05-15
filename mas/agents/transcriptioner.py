"""Voice Transcription Agent"""

import os
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write
from elevenlabs.client import ElevenLabs
from langchain_core.pydantic_v1 import BaseModel, Field

class output_struct_transcriptionagent(BaseModel):
    """Defines the structured output for the transcription result."""
    transcribed_text: str = Field(description="The final transcribed text from the recorded audio.")

class TranscriptionAgent():
    """
    This agent is responsible for recording audio from the microphone, sending it to the ElevenLabs API for transcription, and returning
    the result in a structured JSON format.
    """
    def __init__(self):
        load_dotenv()
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        if not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables.")

        self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)

        self.duration = 10
        self.samplerate = 44100

        self.base_path = os.path.dirname(__file__)
        self.wav_path = os.path.join(self.base_path, "user_request.wav")

    def record_audio(self):
        """
        Records audio from the default microphone and saves it to a WAV file.
        """
        print(f"Recording for {self.duration} seconds...")
        audio_data = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        write(self.wav_path, self.samplerate, audio_data)
        print(f"Recording saved")

    def transcribe_audio(self) -> output_struct_transcriptionagent:
        """
        Transcribes the audio file specified by self.wav_path and returns
        the structured output.
        """
        
        print("Transcribing audio...")
        with open(self.wav_path, "rb") as audio_file:
            response = self.elevenlabs_client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True
            )
        
        
        structured_output = output_struct_transcriptionagent(transcribed_text=response.text)
        
        return structured_output

    def get_transcription_from_mic(self) -> output_struct_transcriptionagent:
        """
        A high-level method that orchestrates the recording and transcription process.
        This is the main method to call from another script.
        
        Returns:
            TranscriptionOutput: A Pydantic object containing the transcribed text.
        """
        self.record_audio()
        transcription_result = self.transcribe_audio()
        return transcription_result
    
    
# if __name__ == "__main__":
#     transcription_agent = TranscriptionAgent()
    
#     final_result = transcription_agent.get_transcription_from_mic()
    
#     print("\n--- Transcription Result (Python Object) ---")
#     print(final_result)
#     print(final_result.transcribed_text)
    
#     # print("\n--- Transcription Result (JSON Format) ---")
#     # print(final_result.json(indent=2))
