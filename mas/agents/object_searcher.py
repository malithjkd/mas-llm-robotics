import os
import cv2
import time
import requests
from dotenv import load_dotenv
import numpy as np
from typing import Optional
from pydantic import BaseModel, Field

class ObjectFindingAgentOutput(BaseModel):
    Object_label: str = Field(description="The label of the object to find (e.g., 'potted plant').")
    Object_Found: str = Field(description="'found' if the object is detected, otherwise 'not found'.")
    Explanation: str = Field(description="Explanation of the detection result.")

class ObjectFindingAgent:
    """
    Agent that uses LandingAI to detect a specific object in a camera frame.
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("LANDING_AI_API_KEY")
        self.url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"

        self.load_additional_info()

    def load_additional_info(self):
        self.base_path = os.path.dirname(__file__)

        self.temp_image_path = os.path.join(self.base_path, "temp_robot_frame.jpg")
        self.output_image_path = os.path.join(self.base_path, "output_object_searcher_frame.jpg")

    def process_user_request(self, image: np.ndarray, object_label: str, draw_boxes: bool = False) -> dict:
        
        cv2.imwrite(self.temp_image_path, image)

        data = {"prompts": object_label, "model": "agentic"}
        headers = {"Authorization": f"Basic {self.api_key}"}

        try:
            with open(self.temp_image_path, "rb") as image_file:
                files = {"image": image_file}
                response = requests.post(self.url, files=files, data=data, headers=headers)
            
            response.raise_for_status() 
            result = response.json()
        except requests.exceptions.RequestException as e:
            print(f"An API request error occurred: {e}")
            return {
                "Object_label": object_label,
                "Object_Found": "error",
                "Explanation": f"Failed to communicate with the API: {e}"
            }
        finally:
            if os.path.exists(self.temp_image_path):
                os.remove(self.temp_image_path)

        found = False
        explanation = ""
        detected_label = object_label

        # Check detection results
        if "data" in result and result["data"] and isinstance(result["data"][0], list):
            for obj in result["data"][0]:
              if obj["score"] > 0.75:
                # if obj["label"].lower() == object_label.lower() and obj["score"] > 0.75:
                    found = True
                    explanation = f"Object '{object_label}' found with confidence {obj['score']:.2f}."
                    if draw_boxes and self.output_image_path:
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, obj['bounding_box'])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, f"{obj['label']} ({obj['score']:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.imwrite(self.output_image_path, image)
                    break

        if not found:
            explanation = f"Object '{object_label}' not found in the image."

        return {
            "Object_label": detected_label,
            "Object_Found": "found" if found else "not found",
            "Explanation": explanation
        }
        

# def find_object_with_scanning(agent: ObjectFindingAgent, robot, object_to_find: str, output_path: Optional[str] = None) -> dict:
#     print(f"Starting a comprehensive search for: '{object_to_find}'")

#     print("Performing initial scan...")
#     frame = robot.get_current_RGB_camera_frames()
#     if frame is None:
#         return {
#             "Object_label": object_to_find,
#             "Object_Found": "error",
#             "Explanation": "Failed to get an image from the camera on the initial attempt."
#         }
    
#     result = agent.process_user_request(frame, object_to_find, draw_boxes=True, output_path=output_path)
    
#     if result.get("Object_Found") == "found":
#         print(f"Success! Object found on the initial scan.")
#         return result

#     # If not found, begin the 360-degree scanning loop
#     print(f"Object not found initially. Beginning 360-degree scan...")
#     max_scans = 8
#     for i in range(max_scans):
#         print(f"--- Scan attempt {i + 1} of {max_scans} ---")
#         robot.rotate_to_scan_environment("left")
#         time.sleep(1)

#         frame = robot.get_current_RGB_camera_frames()
#         if frame is None:
#             print("Warning: Failed to get a frame during this scan step. Continuing...")
#             continue
        
#         result = agent.process_user_request(frame, object_to_find, draw_boxes=True, output_path=output_path)
#         print(f"  - Result: {result['Explanation']}")

#         if result.get("Object_Found") == "found":
#             print(f"Success! Object found after rotation.")
#             return result
    
#     print("Full 360-degree scan complete. Object was not located.")
#     return {
#         "Object_label": object_to_find,
#         "Object_Found": "not found",
#         "Explanation": f"Object '{object_to_find}' was not found after a full 360-degree scan."
#     }

# if __name__ == "__main__":
#     try:
#         agent = ObjectFindingAgent()
#         robot = robot3.Robot()
#         object_to_find = "potted plant"
#         output_img_path = r"D:\University\FYP\FYPGit\MultiAgenticSystemFYP\multiAgenticSystem\MayOOP\multiAgentLLM\Detected_output.png"

#         final_result = find_object_with_scanning(
#             agent=agent, 
#             robot=robot, 
#             object_to_find=object_to_find, 
#             output_path=output_img_path
#         )

        
#         if final_result.get("Object_Found") == "found":
#             print(f"\nTask successful. The object's location is saved in: {output_img_path}")
#         else:
#             print(f"\nTask failed. The object '{object_to_find}' could not be located.")

#     except Exception as e:
#         print(f"A critical error is {e}")
        
    
