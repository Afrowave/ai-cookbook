import json

from ollama import chat, ChatResponse
from pydantic import BaseModel, Field

ai_model = 'qwen3:14b'

# --------------------------------------------------------------
# Define the knowledge base retrieval tool
# --------------------------------------------------------------

def search_data(question: str):
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    with open("data.json", "r") as f:
        return json.load(f)


# --------------------------------------------------------------
# Step 1: Call model with search_data tool defined
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_data",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."
user_prompt= "What is the return policy?"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

response: ChatResponse = chat(
    model=ai_model,
    messages=messages,
    tools=tools,
)


# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------

response.model_dump()

# --------------------------------------------------------------
# Step 3: Execute search_data function
# --------------------------------------------------------------

def call_function(name, args):
    if name == "search_data":
        return search_data(**args)

for tool_call in response.message.tool_calls:
    name = tool_call.function.name
    args = tool_call.function.arguments
    messages.append(response.message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool","tool_call": tool_call.function.name, "content": json.dumps(result)}
    )

# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------

class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")


response_2 = chat(
    model=ai_model,  # Replace with the model you are using
    messages=messages,
    tools=tools,
    format=KBResponse.model_json_schema(),
)

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

final_response = KBResponse.model_validate_json(response_2.message.content)
final_response.answer
final_response.source


# --------------------------------------------------------------
# Question that doesn't trigger the tool
# --------------------------------------------------------------

user_prompt_2 = "What is the weather in Tokyo?"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt_2},
]

response_3= chat(
    model=ai_model,  # Replace with the model you are using
    messages=messages,
    tools=tools,
    format=KBResponse.model_json_schema(),
)

error_response = KBResponse.model_validate_json(response_3.message.content)
error_response.answer
error_response.source
