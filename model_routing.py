import aisuite as ai
import json
from typing import Dict, List, Union
import csv
from prompt_templates import load_privacy_ontology, create_annotation_prompt
from text_processing import process_input

def send_prompt(prompt: str, model_name: str, use_provider: str = "openai", **kwargs):
    """
    Send a prompt to an AI model using aisuite's unified interface
    
    Args:
        prompt (str): The prompt to send
        model_name (str): The full model identifier (e.g., 'openai:gpt-4', 'anthropic:claude-3-5-sonnet')
        use_provider (str): The AI provider to use (default is OpenAI)
        **kwargs: Additional parameters i.e., temperature, max_tokens
    
    Returns:
        str: The AI-generated response
    """
    # Initialize aisuite client
    client = ai.Client()
    
    # Prepare messages in the standard chat format
    messages = [{"role": "system", "content": prompt}]
    
    # Create chat completion
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=kwargs.get('temperature', 0),
        max_tokens=kwargs.get('max_tokens', 1500)
    )
    
    # Extract and return the response content
    return response.choices[0].message.content.strip()

def annotate_with_few_shot_prompt(
    example_directory: str, 
    target_file_path: str, 
    ontology_path: str, 
    model_name: str = "openai:gpt-4", 
    use_provider: str = "openai"
):
    """
    Annotate a file using few-shot prompting with AISuite
    
    Args:
        example_directory (str): Directory containing example files
        target_file_path (str): Path to the target file to annotate
        ontology_path (str): Path to the privacy ontology
        model_name (str): The full model identifier
        use_provider (str): The AI provider to use
    """
    # Load the privacy ontology
    ontology = load_privacy_ontology(ontology_path)
    
    # Load all example files in the directory
    example_files = process_input(example_directory)
    
    # Load the content of the target text file
    with open(target_file_path, 'r') as target_file:
        target_text = target_file.read()
    
    # Generate the few-shot prompt using `create_annotation_prompt`
    prompt = ""
    for example_file in example_files:
        prompt += create_annotation_prompt(example_file, target_text, ontology) + "\n\n"
    
    # Print prompt
    print(prompt)
    
    # Send the prompt to the chosen LLM via AISuite
    annotated_data = send_prompt(
        prompt=prompt,
        model_name=model_name,
        use_provider=use_provider,
        temperature=0,
        max_tokens=1500
    )
    
    print("\nAnnotations from LLM:\n", annotated_data)
    
    # Save the prompt and response to CSV
    with open('llm_output.csv', mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([prompt, annotated_data])

'''
if __name__ == "__main__":
    annotate_with_few_shot_prompt(
        example_directory='./examples', 
        target_file_path='./target.txt', 
        ontology_path='./privacy_ontology.json',
        model_name='openai:gpt-4'
    )
'''