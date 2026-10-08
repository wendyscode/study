# 맹그러봐!!!
# 챗봇 만들기 

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter,TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

#03 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small', 
    api_key= api_key,
    base_url= base_url,
    # dimensions = 5,
)
sample_text = "삼성전자의 창업자는 누구인가요?"
vector = embeddings.embed_query(sample_text)
# print(vector)
print(len(vector))      #1536

DB_PATH = './_db/Chroma12/'

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory= DB_PATH,
    collection_name= 'croma12',
)

print(f"벡터 저장소에 저장된 문서 수: {vector_store._collection.count()}",)
#벡터 저장소에 저장된 문서 수: 59

###################### Retrievers #######################
######################### 검색기 ########################
retriever = vector_store.as_retriever(search_kwargs={"k":2})
print(retriever)


print("======================================================")
###############################모델연결#################################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model= 'gpt-5-nano',
    # model= 'gpt-5.6-terra',
    temperature=0,
    max_tokens = 1000,
    api_key= api_key,
    base_url= base_url,
)


print("======================================================")
print("======================================================")


from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
너는 재미있는 AI야.
너는 '좌좌의 챗봇'이야.

처음 대화를 시작할 때는 반드시
"안녕하세요! 저는 얼렁뚱당 좌좌의 챗봇이에유 😊"
라고 먼저 소개해줘.

그리고마지막에는 
" 제가 알려드리는 정보는 정확하지 않을수 있어유 🙂"라고 마무리해 

어려운 내용도 재미있고 쉽게 설명해줘.
적절하게 이모지도 사용해줘.

반드시 모든 답변을 "~했어유", "~예유", "~해유" 같은 귀엽고 친근한 말투로 작성해줘.
딱딱한 존댓말은 사용하지 말고 자연스럽고 재미있게 말해줘.

답변 300자 넘지마.

컨텍스트 : {context}
질문 : {input}
답변 : 
""")

#체인 만들기 
docu_chain = create_stuff_documents_chain(model, prompt)    #prompt | model
rag_chain = create_retrieval_chain(retriever, docu_chain)   #검색 | docu_chain


############################ Gradio 챗봇 ####################################
############################ Gradio 챗봇 ####################################
import gradio as gr

def answer_invoke(message,history):
    response = rag_chain.invoke({"input" : message}) 
    return response['answer']

# Gradio 인터페이스 만들자 
demo = gr.ChatInterface(fn=answer_invoke, title = 'welcome to 얼렁뚱땅 좌좌 챗봇 0_<')

# Gradio 실행
demo.launch()
# demo.launch(share=True) # 외부에서 접속할수있는 url

#로그확인
# from datetime import datetime

# def answer_invoke(message, history):
#     response = rag_chain.invoke({"input": message})
#     answer = response['answer']

#     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     print("========================================")
#     print("시간 :", now)
#     print("사용자 질문 :", message)
#     print("챗봇 답변 :", answer)
#     print("========================================")

#     return answer

