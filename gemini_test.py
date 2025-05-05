import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

resp = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["What’s 2+2?"]
)

print(resp.text)   
