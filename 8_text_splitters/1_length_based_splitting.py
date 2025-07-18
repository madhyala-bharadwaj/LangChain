from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader("8_text_splitters/dl-curriculum.pdf").load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator="")

result = splitter.split_documents(docs)
print(len(result))
print(result[1])
print(result[1].page_content)
