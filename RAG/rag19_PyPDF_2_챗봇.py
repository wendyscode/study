#챗봇까지  맹그러봐요 
#transformer 논문을 백터디비로 불러와서 
# 요약, 인용 등등 할수있는 챗봇으로!! 

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
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
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)

#백터DB####Vector DB
from langchain_chroma import Chroma

DB_PATH = './_db/attention_paper/'

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='attention_paper',
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

prompt = ChatPromptTemplate.from_template("""
너는 'Attention Is All You Need' 논문을 설명하는 챗봇이다.

아래 논문 컨텍스트를 기반으로 사용자의 질문에 답변한다.

규칙:
- 어려운 내용을 초보자도 이해하기 쉽게 설명한다.
- 논문 내용을 요약할 수 있다.
- 논문의 특정 내용을 질문하면 논문을 근거로 답변한다.
- 논문에 없는 내용은 추측하지 않는다.
- 답변은 500자 이내로 작성한다.
- 자연스러운 존댓말을 사용한다.

논문 컨텍스트:
{context}

질문:
{input}

답변:
""")


from langchain_classic.chains.combine_documents import (create_stuff_documents_chain)
from langchain_classic.chains import (create_retrieval_chain)
docu_chain = create_stuff_documents_chain(model,prompt)
rag_chain = create_retrieval_chain(retriever,docu_chain)

## Gradio 챗봇 ##

import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({
        "input": message
    })
    return response["answer"]

# 챗봇 첫 화면에 초기 안내 메시지 표시
chatbot = gr.Chatbot(
    value=[
        {
            "role": "assistant",
            "content": """안녕하세요! 😊

Attention Is All You Needs 논문 챗봇입니다.

논문의 내용을 질문하거나 요약을 요청해주세요.

예시:
- 논문의 핵심 내용을 요약해주세요.
- Self-Attention이 무엇인가요?
- Multi-Head Attention을 설명해주세요.
- Positional Encoding은 왜 필요한가요?
- Transformer가 RNN보다 좋은 이유는 무엇인가요?"""
        }
    ]
)

demo = gr.ChatInterface(
    fn=answer_invoke,
    chatbot=chatbot,
    title="Attention Is All You Needs 논문 챗봇",
)

#실행 
demo.launch()



