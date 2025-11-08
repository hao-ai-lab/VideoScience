import csv
import sys
import os
import json
import argparse
from google import genai
from google.genai import types
import time
import pandas as pd
from pydantic import BaseModel, Field
sys.path.append(os.getcwd())
from checklist.utils import scrape_text_from_url
from checklist.gemini import GeminiChecklistGenerator

class ChecklistItem(BaseModel):
    phenomenon_congruency: list[str] = Field(default_factory=list)
    correct_dynamism: list[str] = Field(default_factory=list)
    spatio_temporal_continuity: list[str] = Field(default_factory=list)
    immutability: list[str] = Field(default_factory=list)
    interaction_realism: list[str] = Field(default_factory=list)

def create_checklist_prompt(prompt_text, reference_source =""):
    """Create the checklist generation prompt."""
    return f"""From my evaluation of text2video models I have generated a video using the prompt

Prompt: {prompt_text}

{reference_source} 

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



def process_csv_and_generate_checklists(csv_path, output_folder, author, model="models/gemini-2.5-flash"):
    """
    Process CSV file and generate checklists for each row.
    
    Args:
        csv_path: Path to the CSV file
        output_folder: Folder to save the generated checklists
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    gemini = GeminiChecklistGenerator(model=model)
    
    # Read CSV file
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        counter = 0
        for idx, row in enumerate(reader, start=1):
            
            # Get prompt and source
            prompt_text = row.get('Prompts', '').strip()
            source_link = row.get('Source', '').strip()
            id = row.get('Unique ID', '').strip()
            row_author = row.get('Author', '').strip()
            
            if row_author != author:
                print("Author mismatch, skipping row")
                continue
            # Skip if prompt or source is empty
            if not prompt_text or not source_link:
                print(f"Row {idx} ({id}): Skipping - missing prompt or source")
                continue
            
            # Skip if source is not a valid URL
            if not source_link.startswith('http'):
                print(f"Row {idx} ({id}): Skipping - invalid source URL")
                continue

            print(f"\nProcessing row {idx}: {id}")
            print(f"Prompt: {prompt_text[:100]}...")
            print(f"Source: {source_link}")
            
            response_config = {
                        "response_mime_type": "application/json",
                        "response_json_schema": ChecklistItem.model_json_schema(),
                    }
            
            try:
                # Create the prompt
                full_prompt = create_checklist_prompt(prompt_text)
                if "youtube.com" in source_link or "youtu.be" in source_link:
                    response = gemini.generate_from_yt(full_prompt, source_link=source_link, config=response_config)
                else:
                    # Scrape text from URL to use as reference
                    reference_text = scrape_text_from_url(source_link)
                    if reference_text is None:
                        print(f"Row {idx} ({id}): Warning - failed to scrape reference text")
                        continue
                    reference_section = f"Reference Source Text:\n{reference_text}"
                    full_prompt_with_ref = create_checklist_prompt(prompt_text, reference_section)
                    response = gemini.generate(full_prompt_with_ref, config=response_config)
                
                # Rate limiting - wait 1 second between requests
                time.sleep(1)
                
            except Exception as e:
                print(f"✗ Error processing row {idx} (Unique ID {id}): {str(e)}")
                continue
            # Save response to JSON file
            output_path = os.path.join(output_folder, f"checklist_{id}.json")
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(json.loads(response), outfile, indent=4)
            counter += 1
    print(f"\nGenerated {counter} checklists in '{output_folder}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate checklists from CSV prompts.")
    parser.add_argument('--csv_file', type=str, required=False, help='Path to the CSV file containing prompts.')
    parser.add_argument('--output_folder', type=str, required=False, help='Folder to save generated checklists.')
    parser.add_argument('--author', type=str, required=False, help='Author name to filter prompts.')
    parser.add_argument('--model', type=str, default="models/gemini-2.5-flash", help='Gemini model to use for generation.')
    args = parser.parse_args()
    
    # Paths
    csv_file = args.csv_file or "test.csv"
    output_directory = args.output_folder or "generated_checklists"
    author = args.author or "Abhilash"
    output_directory = os.path.join(output_directory, author.replace(" ", "_"))
    
    print("Starting CSV processing...")
    print(f"CSV file: {csv_file}")
    print(f"Output folder: {output_directory}")
    print("="*80)

    process_csv_and_generate_checklists(csv_file, output_directory, author, model=args.model)

    print("\n" + "="*80)
    print("Processing complete!")
