import os
from google import genai

# 1. Instantiate the client with your API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2. Generate a response to “What’s 2+2?”
resp = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["What’s 2+2?"]
)

print(resp.text)   # → "4"
