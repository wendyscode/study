# 맹그러봐 !! 
import os
from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter,TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

#01 데이터를 불러온다.
from langchain_community.document_loaders import PyPDFLoader

path = './_data/'

pdf_loader = PyPDFLoader(
    path + "attention is all you needs.pdf"
)

pdf_docs = pdf_loader.load()

#02 데이터를 자른다. - 문서를 자른다 / 청킹

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(pdf_docs)

#문서 갯수 확인

print("원본 문서 개수 :", len(pdf_docs))
print("청킹 후 문서 개수 :", len(split_docs))

#03. 임베딩 

embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "device": "cpu"
    }
)
#백터DB####Vector DB
from langchain_chroma import Chroma

DB_PATH = './_db/attention_paper_qwen3/'

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='attention_paper_qwen3',
)

if vector_store._collection.count() == 0:
    vector_store.add_documents(split_docs)

print("벡터 DB 문서 개수 :", vector_store._collection.count())

# Retriever 리트리버 #
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

###모델연결#####
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)
#체인 만들기 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system",    """
    
너는 귀여운 챗봇이야.

컨텍스트:
{context}

질문:
{input}

답변:
"""), 
MessagesPlaceholder(variable_name='chat_history'),
("human", "{input}")
])

from langchain_classic.chains.combine_documents import (create_stuff_documents_chain)
from langchain_classic.chains import (create_retrieval_chain)
docu_chain = create_stuff_documents_chain(model,prompt)
rag_chain = create_retrieval_chain(retriever,docu_chain)

## Gradio 챗봇 ##

import gradio as gr

def answer_invoke(message, history):
    chat_history = []


    for item in history[-50:]:
        content = item["content"]

        if item["role"] == "user" : 
            chat_history.append(HumanMessage(content=content))
        elif item["role"] == "assistant":
            chat_history.append(AIMessage(content=content))

    response = rag_chain.invoke({
        "input" : message,
        "chat_history" : chat_history
    })

    return response ["answer"]

# 챗봇 첫 화면에 초기 안내 메시지 표시
chatbot = gr.Chatbot(
    value=[
        {
            "role": "assistant",
            "content": """안녕하세요! 😊
                    귀요미 챗봇입니다.

"""
        }
    ]
)

demo = gr.ChatInterface(
    fn=answer_invoke,
    chatbot=chatbot,
    title=" 귀요미 챗봇",
)

#실행 
demo.launch()


'''
1. 안녕 
history = [
    {"role":"user","content":"안녕"}
    {"role": "assistent", "content":"안녕하세요 " }
]
2. 나는 윤영선이다. 
history = [
    {"role":"user","content":"안녕"}
    {"role": "assistent", "content":"안녕하세요 " }
    {"role":"user","content":"나는 윤영선이다."}
    {"role": "assistent", "content":"당신은 윤영선이시군요 " }
]
'''


