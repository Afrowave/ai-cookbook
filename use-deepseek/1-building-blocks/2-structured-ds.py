from ollama import chat
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
# Step 2: Call the model and use Ollama to get a response
# --------------------------------------------------------------

# Define the prompt to extract event information

system_prompt = "Extract the event information from the following text and return the following keys: 'name', 'date', 'participants'."
user_prompt = "Alice and Bob are going to a science fair on Friday."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]


response = chat(
    model=ai_model,  # Replace with the model you are using
    messages=messages,
    format=CalendarEvent.model_json_schema(),
)

# --------------------------------------------------------------
# Step 3: Parse the JSON response
# --------------------------------------------------------------

event = CalendarEvent.model_validate_json(response.message.content)
event.name
event.date
event.participants
