from dataclasses import Field
import os
import sys
import json
from google import genai
from google.genai import types
sys.path.append(os.getcwd())

class GeminiChecklistGenerator:
    def __init__(self, model='models/gemini-2.5-flash', api_key = None):
        if api_key is not None:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
        self.model = model
        
    def generate(self, prompt_text, config=None):
        response = self.client.models.generate_content(
            model=self.model,
            contents=types.Content(
                parts=[
                    types.Part(text=prompt_text)
                ]
            ),
            config=config
        )
        return response.text
    def generate_from_yt(self, prompt_text, source_link, config=None):
        """Generate checklist from youtube video using Gemini API."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(file_uri=source_link),
                    ),
                    types.Part(text=prompt_text)
                ]
            ),
            config=config
        )
        return response.text
if __name__ == "__main__":
    gemini = GeminiChecklistGenerator()

    text = "A raw egg lies on a smooth flat table surface. A hand spins the egg rapidly, causing it to rotate at high speed. After the egg has been spinning for 2-3 seconds, a finger reaches in and briefly presses down on the top of the egg to stop its rotation completely. The finger is held there for about one second, then quickly lifted away."
    video_reference = 'https://youtu.be/_G5zNZf_6g4?si=k6Vopjk0Du-VLt_w'

    prompt = """From my evaluation of text2video models I have generated a video using the prompt

    Prompt: {text}
    
    {reference_text}

    Use reference source to create the checklist, use the youtube video (mandatory)

    Create an expected phenomenon checklist targeting the following categories (note: not all categories are required for a prompt)

    1. PHENOMENON CONGRUENCY: Does the video show the correct expected phenomenon?

    2. CORRECT DYNAMISM Are the physics dynamics and motion behaviors accurate?

    3. SPATIO-TEMPORAL CONTINUITY Are spatial relationships and temporal sequences physically consistent?

    4. IMMUTABILITY Do object properties remain physically consistent?

    5.  INTERACTION REALISM Do object interactions follow physical laws?

    Guidelines for checklist creation:
    - only target things which are visually observable in the video
    - the statements in checklist needs to be assertive statements instead of questions
    """
    from pydantic import BaseModel, Field
    class ChecklistItem(BaseModel):
        phenomenon_congruency: list[str] = Field(default_factory=list)
        correct_dynamism: list[str] = Field(default_factory=list)
        spatio_temporal_continuity: list[str] = Field(default_factory=list)
        immutability: list[str] = Field(default_factory=list)
        interaction_realism: list[str] = Field(default_factory=list)
    config = {
        "response_mime_type": "application/json",
        "response_json_schema": ChecklistItem.model_json_schema(),
    }   
    response = gemini.generate_from_yt(prompt.format(text=text, reference_text = ""), source_link=video_reference, config=config)
    print(json.loads(response))
    # print(ChecklistItem.model_validate_json(response))
    
    text = "A clear glass beaker filled with still, cold water sits on a white surface. A dropper containing blue food coloring is held with its tip positioned just above the water surface. A single drop of blue food coloring is slowly released from the dropper into the water."
    url = "https://www.sciencing.com/happens-food-coloring-cold-water-8253853"
    from checklist.utils import scrape_text_from_url
    reference_text = scrape_text_from_url(url)
    if reference_text is None:
        raise ValueError("Failed to scrape text from the reference URL.")

    response = gemini.generate(prompt.format(text=text, reference_text=f"Reference Source Text:\n{reference_text}"), config=config)
    print(json.loads(response))
    