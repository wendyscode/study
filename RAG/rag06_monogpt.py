from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"
llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra', 
    temperature= 0,
    api_key= api_key,
    base_url= base_url,
)

# response = llm.invoke('나는 윤영선이야. 나는 잘생겼지. 이해했니?')
response = llm.invoke('나는 누구게?')

# print(response)
print(response.content)


