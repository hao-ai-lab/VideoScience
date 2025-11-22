import csv
import sys
import os
import json
import argparse
from google import genai
from google.genai import types
import time
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from checklist.utils import scrape_text_from_url
from checklist.gemini import GeminiChecklistGenerator


class ChecklistItem(BaseModel):
    """
    This schema matches what vlm_as_a_judge.py expects:
      {
        "phenomenon_congruency": [...],
        "correct_dynamism": [...],
        "spatio_temporal_continuity": [...],
        "immutability": [...],
        "interaction_realism": [...]
      }
    """
    phenomenon_congruency: list[str] = Field(default_factory=list)
    correct_dynamism: list[str] = Field(default_factory=list)
    spatio_temporal_continuity: list[str] = Field(default_factory=list)
    immutability: list[str] = Field(default_factory=list)
    interaction_realism: list[str] = Field(default_factory=list)


def create_checklist_prompt(
    prompt_text: str,
    reference_phenomenon: str = "",
    additional_reference_from_multimedia: str = "",
    source_link: str = "",
) -> str:
    """Create the checklist generation prompt."""
    return f"""From my evaluation of text2video models I have generated a video using the prompt:

Prompt: {prompt_text}

Reference ground-truth phenomenon (expected outcome / behavior):
{reference_phenomenon}

Additional information about the multimedia or reference sources:
{additional_reference_from_multimedia}

Reference video or page URL (if available):
{source_link}

Use these reference sources to create the checklist (and the reference video if available).

Create a checklist targeting each of the following categories (note: not all categories are required for a prompt):

1. PHENOMENON CONGRUENCY
   - Does the video show the correct expected phenomenon?

2. CORRECT DYNAMISM
   - Are the physics dynamics and motion behaviors accurate?

3. SPATIO-TEMPORAL CONTINUITY
   - Are spatial relationships and temporal sequences physically consistent?

4. IMMUTABILITY
   - Do object properties remain physically consistent?

5. INTERACTION REALISM
   - Do object interactions follow physical laws (including those not explicitly described in the expected phenomenon)?

Guidelines for checklist creation:
- Only target things which are visually observable in the video.
- The statements in the checklist must be assertive statements that can be answered with YES or NO (not questions).
- Each statement should be as concrete and testable as possible.
"""


def process_csv_and_generate_checklists(
    csv_path: str,
    output_folder: str,
    author: str,
    model: str = "models/gemini-2.5-flash",
    max_entries: int = 0,
) -> None:
    """
    Process CSV file and generate checklists for each row.

    Args:
        csv_path: Path to the CSV file
        output_folder: Folder to save the generated checklists
        author: Author name to filter rows
        model: Gemini model name
        max_entries: Max number of checklists to generate (0 = no limit)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generator client
    gemini = GeminiChecklistGenerator(model=model)

    # Read CSV file
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        counter = 0  # number of successfully generated checklists

        for idx, row in enumerate(reader, start=1):
            # If we already generated max_entries, stop
            if max_entries > 0 and counter >= max_entries:
                print(f"Reached max_entries={max_entries}, stopping.")
                break

            # Get prompt and source
            prompt_text = (row.get("Prompts") or "").strip()
            reference_phenomenon = (row.get("Expected phenomenon") or "").strip()
            source_link = (row.get("Source") or "").strip()
            uid = (row.get("Unique ID") or "").strip()
            row_author = (row.get("Author") or "").strip()

            if row_author != author:
                print(f"Row {idx} ({uid}): Author mismatch ({row_author} != {author}), skipping.")
                continue

            # Skip if prompt or source is empty
            if not prompt_text or not source_link:
                print(f"Row {idx} ({uid}): Skipping - missing prompt or source")
                continue

            # Skip if source is not a valid URL
            if not source_link.startswith("http"):
                print(f"Row {idx} ({uid}): Skipping - invalid source URL: {source_link}")
                continue

            print(f"\nProcessing row {idx}: {uid}")
            #print(f"  Prompt: {prompt_text[:100]}...")
            print(f"  Prompt: {prompt_text}")
            print(f"  Source: {source_link}")

            response_config = {
                "response_mime_type": "application/json",
                "response_json_schema": ChecklistItem.model_json_schema(),
            }


            # Build prompt differently depending on whether it's YouTube or not
            if "youtube.com" in source_link or "youtu.be" in source_link:
                full_prompt = create_checklist_prompt(
                    prompt_text=prompt_text,
                    reference_phenomenon=reference_phenomenon,
                    additional_reference_from_multimedia="Reference is a YouTube video containing the expected phenomenon.",
                    source_link=source_link,
                )
                raw_response = gemini.generate_from_yt(
                    full_prompt,
                    source_link=source_link,
                    config=response_config,
                )
            else:
                # Scrape text from URL to use as reference
                reference_text = scrape_text_from_url(source_link)
                if reference_text is None:
                    print(f"Row {idx} ({uid}): Warning - failed to scrape reference text, skipping.")
                    continue

                reference_section = f"Reference Source Text:\n{reference_text}"
                full_prompt = create_checklist_prompt(
                    prompt_text=prompt_text,
                    reference_phenomenon=reference_phenomenon,
                    additional_reference_from_multimedia=reference_section,
                    source_link=source_link,
                )

                raw_response = gemini.generate(
                    full_prompt,
                    config=response_config,
                )

            # raw_response is expected to be a JSON string; validate and normalize via Pydantic
            try:
                checklist_obj = ChecklistItem.model_validate_json(raw_response)
            except ValidationError as ve:
                print(f"Row {idx} ({uid}): Validation error for checklist JSON: {ve}")
                continue

            checklist_dict = checklist_obj.model_dump()

            # Save response to JSON file – this shape is directly consumable by vlm_as_a_judge.py
            output_path = os.path.join(output_folder, f"checklist_{uid}.json")
            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(checklist_dict, outfile, indent=4, ensure_ascii=False)

            print(f"Saved checklist JSON to: {output_path}")
            counter += 1

            # Rate limiting - wait 1 second between requests
            time.sleep(1)

            print(f"Error processing row {idx} (Unique ID {uid})")
            continue

    print(f"\nGenerated {counter} checklists in '{output_folder}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate checklists from CSV prompts.")
    parser.add_argument(
        "--csv_file",
        type=str,
        required=False,
        help="Path to the CSV file containing prompts.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        help="Folder to save generated checklists.",
    )
    parser.add_argument(
        "--author",
        type=str,
        required=False,
        help="Author name to filter prompts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gemini-flash-latest",
        help="Gemini model to use for generation.",
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=0,
        help="Maximum number of checklists to generate (0 = no limit).",
    )
    args = parser.parse_args()

    # Paths
    csv_file = args.csv_file or "test.csv"
    output_root = args.output_folder or "generated_checklists"
    author = args.author or "Abhilash"

    # Match other scripts: put author in a subfolder with spaces replaced by underscores
    author_subdir = author.replace(" ", "_")
    output_directory = os.path.join(output_root, author_subdir)

    print("Starting CSV processing...")
    print(f"CSV file:      {csv_file}")
    print(f"Output folder: {output_directory}")
    print(f"Author:        {author}")
    print(f"Max entries:   {args.max_entries}")
    print("=" * 80)

    process_csv_and_generate_checklists(
        csv_path=csv_file,
        output_folder=output_directory,
        author=author,
        model=args.model,
        max_entries=args.max_entries,
    )

    print("\n" + "=" * 80)
    print("Processing complete!")
