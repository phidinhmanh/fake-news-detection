import os
import sys
from google import genai
from dotenv import load_dotenv

# Fix path to allow importing sequential_adversarial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'} (starts with {api_key[:5] if api_key else 'None'})")

client = genai.Client(api_key=api_key)

try:
    print("Available models:")
    for m in client.models.list():
        print(m.name)
            
    # Try gemini-flash-latest from the user's list
    model_id = "gemini-flash-latest" 
    response = client.models.generate_content(model=model_id, contents="Hello! Are you working?")
    print(f"Success with {model_id}!")
    print(response.text)
except Exception as e:
    print(f"Error calling gemini: {e}")

from sequential_adversarial.llm_client import LLMClient
llm = LLMClient(mock=False)
try:
    print(f"LLMClient configured model name: {llm.model_name}")
    resp = llm.generate("Hello, from LLMClient directly.")
    print("LLMClient direct result:")
    print(resp[:200])
except Exception as e:
    print(f"LLMClient error: {e}")
