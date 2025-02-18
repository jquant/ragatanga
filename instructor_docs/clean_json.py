import json
import os
from pathlib import Path

def clean_json_files(input_folder):
    # Create output folder if it doesn't exist
    input_path = Path(input_folder)
    output_folder = input_path.parent / f"{input_path.name}_clean"
    output_folder.mkdir(exist_ok=True)
    
    # Process all JSON files in the input folder
    for json_file in input_path.glob('*.json'):
        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # For each item in the array, keep only the markdown field
            cleaned_data = [{'content': item['markdown']} for item in data]
            
            # Create output file path
            output_file = output_folder / json_file.name
            
            # Save the cleaned JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
                
            print(f"Cleaned {json_file.name}")
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")

if __name__ == "__main__":
    # Get input folder from user
    folder_path = input("Enter the folder path containing JSON files: ")
    
    if os.path.exists(folder_path):
        clean_json_files(folder_path)
        print("Cleaning complete!")
    else:
        print("Invalid folder path!") 