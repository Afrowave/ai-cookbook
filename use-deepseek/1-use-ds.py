from ollama import chat
from ollama import ChatResponse

ai_model = 'deepseek-r1:8b'

response: ChatResponse =chat(model=ai_model, messages=[{'role': 'user', 'content': 'Tell me a joke.'}])

# print(response['message']['content'])
print(response.message.content)
