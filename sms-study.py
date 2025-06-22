from langfuse import Langfuse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

# Load environment variables from .env file
load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

langfuse_handler = CallbackHandler()

# Initialize Google model
llm = ChatGoogleGenerativeAI(
    model="gemma-3n-e4b-it",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Simple test call with Langfuse tracing
def test_llm_call():
    try:
        # Create a simple message
        message = HumanMessage(content="explica la computacion cuantica en una frase")
        
        # Make the LLM call with Langfuse tracing
        response = llm.invoke([message], config={"callbacks": [langfuse_handler]})
        
        print("LLM Response:")
        print(response.content)
        print("\nTracing enabled - check your Langfuse dashboard!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm_call()




