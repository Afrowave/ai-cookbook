from ollama import chat
from ollama import ChatResponse

ai_model = 'qwen2.5-coder:14b'

response: ChatResponse =chat(model=ai_model, messages=[{'role': 'user',
    'content': 'Tell me a joke.'}])

print(response['message']['content'])
print(response.message.content)
