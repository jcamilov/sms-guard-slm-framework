from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import time
import re
from prompts import prompts

load_dotenv()

# --- Langfuse and LLM Initialization ---

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

model_name = "gemma-3n-e4b-it"
llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
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

# --- Main Evaluation Logic ---

def run_evaluation():
    """
    Runs the evaluation of SMS classification prompts against a Langfuse dataset.
    """
    dataset_name = "smallTestSet"  

    print(f"Fetching dataset '{dataset_name}' from Langfuse...")
    try:
        dataset = langfuse.get_dataset(dataset_name)
    except Exception as e:
        print(f"🚨 Error fetching dataset: {e}")
        print("Please ensure the dataset name is correct and it exists in your Langfuse project.")
        return

    print("Starting evaluation...")
    # Iterate over each prompt from propmts.py
    for prompt_name, prompt_data in prompts.items():
        print(f"\n--- Evaluating Prompt: {prompt_name} ('{prompt_data['description']}') ---")

        # The handler is stateless, it only needs to be initialized once
        handler = CallbackHandler()

        # Iterate over each item in the dataset
        for item in dataset.items:
            # item.run() creates a trace and links it to the dataset item.
            # It requires `run_name` to group evaluations.
            # `run_metadata` is passed to add custom metadata to the run item.
            with item.run(
                run_name=prompt_name,
                tags=[prompt_name, model_name]
            ) as run:
                # Format the prompt with the SMS text from the dataset item
                formatted_prompt = prompt_data["prompt"].format(sms_text=item.input)
                message = HumanMessage(content=formatted_prompt)

                retries = 3
                delay = 2  # seconds
                
                for attempt in range(retries):
                    try:
                        # Make the LLM call with the handler
                        response = llm.invoke(
                            [message],
                            config={"callbacks": [handler]}
                        )
                        
                        # Parse the output
                        parsed_output = parse_classification(response.content)

                        # Map numeric expected output to string classification
                        # "1" is considered "smishing", anything else is "benign"
                        expected_classification = "smishing" if item.expected_output == "1" else "benign"

                        # Score the run directly
                        run.score(
                            name="classification-accuracy",
                            value=1 if parsed_output == expected_classification else 0,
                            comment=f"Model classified as {parsed_output}, expected {expected_classification}"
                        )

                        print(f"  - SMS: '{item.input[:30]}...' -> Classified as: {parsed_output} (Expected: {expected_classification})")
                        
                        break  # Success, exit retry loop

                    except Exception as e:
                        # Handle potential rate limiting or other API errors
                        print(f"  - ⚠️ Attempt {attempt + 1} failed: {e}")
                        if attempt < retries - 1:
                            print(f"    Retrying in {delay} seconds...")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                        else:
                            print("  - ❌ Max retries reached. Moving to next item.")


    print("\n✅ Evaluation complete!")
    print("Check your Langfuse dashboard to see the results and compare prompt performance.")


if __name__ == "__main__":
    run_evaluation()




