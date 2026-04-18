import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'} (starts with {api_key[:5] if api_key else 'None'})")

genai.configure(api_key=api_key)

try:
    print("Available models:")
    models = list(genai.list_models())
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
            
    # Try gemini-pro which is generally more stable across key boundaries
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content("Hello! Are you working?")
    print("Success with gemini-pro!")
    print(response.text)
except Exception as e:
    print(f"Error calling gemini: {e}")

from sequential_adversarial.llm_client import LLMClient
client = LLMClient(mock=False)
try:
    print(f"LLMClient configured model name: {client.model_name}")
    resp = client.generate("Hello, from LLMClient directly.")
    print("LLMClient direct result:")
    print(resp[:200])
except Exception as e:
    print(f"LLMClient error: {e}")
