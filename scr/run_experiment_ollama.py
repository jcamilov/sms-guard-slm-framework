from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import time
import re
import csv
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.prompts import prompts
from datasets.small_test_dataset import small_test_dataset


load_dotenv()

# --- LLM Initialization ---

model_name = "gemma3n:e2b"  # You can change this to any model available in your Ollama installation
llm = Ollama(
    model=model_name,
    temperature=0.7
)

# --- Helper Functions ---

def parse_classification(response_text):
    """
    Parses the LLM response to extract the classification.
    """
    match = re.search(r"##Classification:\s*['\"]?(smishing|benign)['\"]?", response_text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "unclassified"

def parse_explanation(response_text):
    """
    Parses the LLM response to extract the explanation.
    """
    match = re.search(r"##Explanation:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No explanation provided"

def save_results_to_csv(results, filename=None):
    """
    Saves the results to a CSV file for easy Excel import.
    """
    # Create experiment_results directory if it doesn't exist
    experiment_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_results")
    os.makedirs(experiment_results_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ollama_sms_experiment_results_{timestamp}.csv"
    
    # Create full path in experiment_results folder
    filepath = os.path.join(experiment_results_dir, filename)
    
    fieldnames = [
        'model_name',
        'timestamp',
        'prompt_name', 
        'sms_id', 
        'sms_text', 
        'original_classification', 
        'model_classification', 
        'explanation',
        'is_correct'
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    return filepath

# --- Main Evaluation Logic ---

def run_evaluation():
    """
    Runs the evaluation of SMS classification prompts against the local dataset using Ollama.
    """
    print("Starting SMS classification experiment with Ollama...")
    print(f"Model: {model_name}")
    print(f"Dataset size: {len(small_test_dataset)} SMS messages")
    print(f"Number of prompts: {len(prompts)}")
    
    results = []
    total_runs = len(prompts) * len(small_test_dataset)
    current_run = 0
    
    # Iterate over each prompt from prompts.py
    for prompt_name, prompt_data in prompts.items():
        print(f"\n--- Evaluating Prompt: {prompt_name} ('{prompt_data['description']}') ---")

        # Iterate over each item in the dataset
        for sms_item in small_test_dataset:
            current_run += 1
            print(f"  Progress: {current_run}/{total_runs} - SMS ID: {sms_item['sms_id']}")
            
            # Format the prompt with the SMS text from the dataset item
            formatted_prompt = prompt_data["prompt"].format(sms_text=sms_item['sms_text'])
            message = HumanMessage(content=formatted_prompt)

            retries = 3
            delay = 1  # seconds (Ollama is typically faster than cloud APIs)
            
            for attempt in range(retries):
                try:
                    # Make the LLM call
                    response = llm.invoke([message])
                    
                    # Parse the output - Ollama returns string directly
                    response_text = response if isinstance(response, str) else str(response)
                    parsed_classification = parse_classification(response_text)
                    parsed_explanation = parse_explanation(response_text)
                    
                    # Determine if classification is correct
                    is_correct = parsed_classification == sms_item['class']
                    
                    # Create result record
                    result_record = {
                        'model_name': model_name,
                        'timestamp': datetime.now().isoformat(),
                        'prompt_name': prompt_name,
                        'sms_id': sms_item['sms_id'],
                        'sms_text': sms_item['sms_text'],
                        'original_classification': sms_item['class'],
                        'model_classification': parsed_classification,
                        'explanation': parsed_explanation,
                        'is_correct': is_correct
                    }
                    
                    results.append(result_record)
                    
                    # Print result
                    status = "✅" if is_correct else "❌"
                    print(f"    {status} Classified as: {parsed_classification} (Expected: {sms_item['class']})")
                    
                    break  # Success, exit retry loop

                except Exception as e:
                    # Handle potential connection errors or other issues
                    print(f"    ⚠️ Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        print(f"      Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        print("    ❌ Max retries reached. Moving to next item.")
                        # Add failed record
                        result_record = {
                            'model_name': model_name,
                            'timestamp': datetime.now().isoformat(),
                            'prompt_name': prompt_name,
                            'sms_id': sms_item['sms_id'],
                            'sms_text': sms_item['sms_text'],
                            'original_classification': sms_item['class'],
                            'model_classification': 'ERROR',
                            'explanation': f'Error: {str(e)}',
                            'is_correct': False
                        }
                        results.append(result_record)

    # Save results to CSV
    filename = save_results_to_csv(results)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    # Calculate overall accuracy
    successful_runs = [r for r in results if r['model_classification'] != 'ERROR']
    correct_classifications = [r for r in successful_runs if r['is_correct']]
    
    overall_accuracy = len(correct_classifications) / len(successful_runs) * 100 if successful_runs else 0
    error_rate = (len(results) - len(successful_runs)) / len(results) * 100
    
    print(f"Total runs: {len(results)}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Failed runs: {len(results) - len(successful_runs)}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print(f"Error rate: {error_rate:.2f}%")
    
    # Per-prompt statistics
    print("\nPer-prompt statistics:")
    for prompt_name in prompts.keys():
        prompt_results = [r for r in results if r['prompt_name'] == prompt_name and r['model_classification'] != 'ERROR']
        if prompt_results:
            prompt_correct = [r for r in prompt_results if r['is_correct']]
            prompt_accuracy = len(prompt_correct) / len(prompt_results) * 100
            print(f"  {prompt_name}: {prompt_accuracy:.2f}% ({len(prompt_correct)}/{len(prompt_results)})")
    
    print(f"\n✅ Results saved to: {filename}")
    print("You can now import this CSV file into Excel for detailed analysis.")

if __name__ == "__main__":
    run_evaluation() 