import ollama
import json

from pydantic import BaseModel

ai_model = 'deepseek-r1:8b'

# --------------------------------------------------------------
# Step 1: Define the response format in a Pydantic model
# --------------------------------------------------------------

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# --------------------------------------------------------------
# Step 2: Call the model and use Ollama to generate the response
# --------------------------------------------------------------

# Define the prompt to extract event information
prompt = """
Extract the event information from the following text and return it into JSON object with the following keys: "name", "date", "participants" without the markdown format.

Text: "Alice and Bob are going to a science fair on Friday."
"""

response = ollama.generate(
    model=ai_model,  # Replace with the model you are using
    prompt=prompt,
)

generated_content = response['response']


# --------------------------------------------------------------
# Step 3:  Remove content between <think> and </think> tags and Markdown format
# --------------------------------------------------------------

clean_response = generated_content.split("</think>")[-1].strip()
clean_json = clean_response.strip().replace("```json", "").replace("```", "").strip()

# --------------------------------------------------------------
# Step 4:# Parse the JSON response
# --------------------------------------------------------------

try:
    event_data = json.loads(clean_json)
    event = CalendarEvent(**event_data)
    event.name
    event.date
    event.participants
    print(event)
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")
