from langchain_community.document_loaders import TextLoader, PyPDFLoader

path = './_data/'
pdf_loader  = PyPDFLoader(path + "attention is all you needs.pdf")
pdf_docs = pdf_loader.load()

print(type(pdf_docs))
print(len(pdf_docs))
# print(pdf_docs)

print("=====================================================")
print(pdf_docs[0])
#[0] → 1페이지  /[1] → 2페이지
print("=====================================================")
