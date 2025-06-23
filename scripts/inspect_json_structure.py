import json
from pathlib import Path

def inspect_json_file(json_path: Path):
    """
    Loads the specified JSON file and provides a transparent inspection of the
    'words' list to verify its structure and count.
    """
    print(f"--- Inspecting structure of: {json_path} ---")
    
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'words' not in data:
        print("Error: The JSON file does not have a top-level 'words' key.")
        return

    words_list = data['words']

    print(f"\\n1. The type of the value at the 'words' key is: {type(words_list)}")

    if not isinstance(words_list, list):
        print("Error: The value at the 'words' key is not a list as expected.")
        return

    word_count = len(words_list)
    print(f"2. The total number of items in this list is: {word_count:,}")

    print("\\n3. The first 5 items in the list are:")
    for i, item in enumerate(words_list[:5]):
        print(f"   Item {i}: {item}")

    print("\\n4. The last 5 items in the list are:")
    for i, item in enumerate(words_list[-5:]):
        # Calculate the actual index for display
        actual_index = word_count - 5 + i
        print(f"   Item {actual_index}: {item}")
    
    print("\\n" + "="*50)
    print("Conclusion: The word count is the length of this list.")
    print("="*50)


def main():
    """
    Main function to execute the inspection.
    """
    json_file_path = Path('data/processed/full_podcast/elevenlabs_transcript.json')
    inspect_json_file(json_file_path)


if __name__ == "__main__":
    main() 