import json
import re

import ollama
from pydantic import BaseModel

ai_model = 'qwen2.5-coder:14b'

# --------------------------------------------------------------
# Step 1: Define the response format in a Pydantic model
# --------------------------------------------------------------

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# --------------------------------------------------------------
# Step 2: Call the model and ask it to respond in JSON
# --------------------------------------------------------------

# Define the prompt to extract event information
prompt = """
System: Extract the event information from the following text and return the following keys: "name", "date", "participants". Response using JSON.

Text: "Alice and Bob are going to a science fair on Friday."
"""

response = ollama.generate(
    model=ai_model,  # Replace with the model you are using
    prompt=prompt
)

generated_content = response['response']
print(generated_content)


# --------------------------------------------------------------
# Step 3: Parse the JSON response
# --------------------------------------------------------------

try:
    event_data = json.loads(generated_content)
    event = CalendarEvent(**event_data)
    event.name
    event.date
    event.participants
    print(event)
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")
