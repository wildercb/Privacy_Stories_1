import os
import re

def clean_full_text(text):
    """Removes annotations (A:, DT:, P:), section tags {#s ... \}, and <PI> tags from the text."""
    cleaned_text = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)', '', text)
    cleaned_text = re.sub(r'{#s|\}', '', cleaned_text)
    cleaned_text = re.sub(r'<PI>', '', cleaned_text)
    return cleaned_text.strip()

def process_file(file_path):
    """Processes a single file to extract cleaned full text and sections with metadata."""
    with open(file_path, 'r') as file:
        text = file.read()

    # Full cleaned text without annotations or section tags
    cleaned_full_text = clean_full_text(text)
    
    # Prepare dictionary structure
    result = {
        "file_name": os.path.basename(file_path),
        "full_cleaned_text": cleaned_full_text,
        "sections": []
    }

    # Extract sections defined by {#s ... \}
    section_pattern = re.compile(r'{#s(.*?)\\}', re.DOTALL)
    sections = section_pattern.findall(text)

    for section in sections:
        # Clean section text by removing annotations
        cleaned_section = re.sub(r'\(A:.*?\)|\(DT:.*?\)|\(P:.*?\)', '', section).strip()

        # Extract metadata (actions, data types, purposes)
        actions = re.findall(r'\(A:\s*(.*?)\)', section)
        data_types = re.findall(r'\(DT:\s*(.*?)\)', section)
        purposes = re.findall(r'\(P:\s*(.*?)\)', section)
        
        # Add section data to dictionary
        result["sections"].append({
            "section_text_with_tags": cleaned_section,
            "cleaned_section_text": cleaned_section,
            "metadata": {
                "actions": actions if actions else None,
                "data_types": data_types if data_types else None,
                "purposes": purposes if purposes else None
            }
        })
        
    return result

def process_directory(directory_path):
    """Processes all text files in a directory and returns a list of dictionaries for each file."""
    results = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            file_data = process_file(file_path)
            results.append(file_data)
    return results

def process_input(path):
    """Determines if the path is a file or directory and processes accordingly."""
    if os.path.isdir(path):
        # Process all files in the directory
        return process_directory(path)
    elif os.path.isfile(path):
        # Process a single file
        return [process_file(path)]
    else:
        raise ValueError("Invalid path. Please provide a valid file or directory path.")
