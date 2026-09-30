from langchain_openai import ChatOpenAI
import os
# os.environ["OPENAI_API_KEY"] ="🍧🍧🍧key자리🍧🍧🍧"

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra', 
    temperature= 0,
    # openai_api_key = openai_api_key,
)

# response = llm.invoke('나는 윤영선이야. 나는 잘생겼지. 이해했니?')
response = llm.invoke('나는 누구게?')

# print(response)
print(response.content)