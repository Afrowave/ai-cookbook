from ollama import chat
from ollama import ChatResponse

ai_model = 'qwen3:14b'

response: ChatResponse = chat(model=ai_model, messages=[{'role': 'user',
                                                        'content': 'Tell me a joke.'}])

print(response.message.content)
# The above code initializes a connection to an AI model named 'qwen3:14b' using the ollama library.
# It sends a message to the AI model asking for a joke.
# The response from the AI model is stored in the variable 'response'.
# Finally, it prints the content of the response message.

# To explain the code further:
# 1. `from ollama import chat` and `from ollama import ChatResponse` import necessary functions and classes from the ollama library.
# 2. `ai_model = 'qwen3:14b'` sets the AI model to be used for generating responses.
# 3. The `chat` function is called with the specified model and a list of messages as arguments.
#    Each message in the list is a dictionary with two keys: 'role' (which can be 'user', 'assistant', etc.) and 'content' (the actual text of the message).
# 4. The response from the `chat` function is stored in the variable `response`.
# 5. Finally, `print(response.message.content)` prints the content of the response message.oll
