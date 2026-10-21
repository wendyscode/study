# pip install rank_bm25

from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = "삼성전자의 반도체 사업"

# ###################### BM25 검색 ##########################
# # BM25는 키워드 기반 검색. Best Match 25
# from langchain_community.retrievers import BM25Retriever

# bm25_retriever = BM25Retriever.from_documents(texts)
# bm25_retriever.k = 3

# bm25_docs = bm25_retriever.invoke(query)

# print("===================== BM25 검색 결과 =========================")
# for doc in bm25_docs:
#     print(doc.page_content)

######################################################################
######################################################################
######################################################################
########## 벡터 검색 ##########

import os
from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter,TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()
# api_key = os.environ["MONOROUTER_API_KEY"].strip()
# base_url = "https://monogpt.kr/api/monorouter/v1"

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

DB_PATH = './_db/Chroma12/'

vector_store = Chroma.from_documents(
    embedding=embeddings,
    documents=split_docs,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)

vector_retriver = vector_store.as_retriever(
    search_kwargs={"k":3}
)
vector_docs = vector_retriver.invoke(query)

print("====================벡터 검색결과 =====================")
for doc in vector_docs:
    print(doc.page_content)

