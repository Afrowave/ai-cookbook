import json
import requests

from ollama import chat, ChatResponse
from pydantic import BaseModel, Field


ai_model = 'qwen2.5-coder:14b'

# --------------------------------------------------------------
# Step 1: Define the tool (function) that we want to call
# --------------------------------------------------------------

def get_weather(latitude, longitude):
    """
    This is a publically available API that returns the weather for a given location. This is an arbitrary API endpoint which can be any source of data.
    """
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


# --------------------------------------------------------------
# Step 2: Define the get_weather tool and LLM parameters
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful weather assistant."
user_prompt = "What's the weather like in Nairobi today?"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

# --------------------------------------------------------------
# Step 3: Call the model and use Ollama to get the response
# --------------------------------------------------------------

response: ChatResponse = chat(
    model=ai_model,
    messages=messages,
    tools=tools,
)

response.model_dump()

# --------------------------------------------------------------
# Step 4: Execute get_weather function
# --------------------------------------------------------------

def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)

for tool_call in response.message.tool_calls:
    name = tool_call.function.name
    args = tool_call.function.arguments
    messages.append(response.message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call": tool_call.function.name, "content": json.dumps(result)}
    )

# --------------------------------------------------------------
# Step 5: Supply result and call model again
# --------------------------------------------------------------


class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )


response_2 = chat(
    model=ai_model,  # Replace with the model you are using
    messages=messages,
    tools=tools,
    format=WeatherResponse.model_json_schema(),
)

# --------------------------------------------------------------
# Step 5: Produce model response in the WeatherResponse format
# --------------------------------------------------------------

weather = WeatherResponse.model_validate_json(response_2.message.content)
weather.temperature
weather.response
