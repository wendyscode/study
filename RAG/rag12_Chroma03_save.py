import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter,TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

from glob import glob

path = './_data/rag_data/'
#폴더에서 텍스트 파일 목록 가져오기 
txt_file = glob(os.path.join(path, '*txt'))
print(txt_file)
# ['./_data/rag_data\\2026_AI_for_All.txt', 
# './_data/rag_data\\nvidia_outlook.txt', 
# './_data/rag_data\\samsung_outlook.txt']

#01 데이터를 불러온다.
data = []
for text_file in txt_file: 
    loader = TextLoader(text_file, encoding='utf-8')
    # data + loader
    data += loader.load()
print(data)

print("=============================")
#print(data[0])
print("=============================")
print(len(data))
print(data[0].page_content)

char_count = [len(doc.page_content) for doc in data]
print(char_count)   #[8158, 2049, 1898]

#02 데이터를 자른다.(청킹)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size =300,
    chunk_overlap = 100,
    separators=["\n\n","\n"," ",""]     #통상 디폴트.
)

texts = text_splitter.split_documents(data)
print("생성된 텍스트 청크수 :", len(texts))
print("각 청크의 길이  :", list(len(text.page_content)for text in texts))



exit()

#문서를 자른다 / 청킹
split_doc1 = loader1.load_and_split(text_splitter)   #청크300, 오버랩100
split_doc2 = loader2.load_and_split(text_splitter)   #청크300, 오버랩100

#문서 갯수 확인
print(split_doc1)
print(len(split_doc1),len(split_doc2))  # 9 9

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small', 
    api_key= api_key,
    base_url= base_url,
    # dimensions = 5,
)

DB_PATH = './_db/Chroma11/'
#저장
db = Chroma.from_documents(
    documents= split_doc1 + split_doc2,
    embedding=embeddings,
    persist_directory= DB_PATH,
    collection_name= 'croma11',
)

print("Chroma 문서저장 끗 ")