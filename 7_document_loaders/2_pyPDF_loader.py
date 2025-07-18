from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("7_document_loaders\\dl-curriculum.pdf")
docs = loader.load()

print(len(docs))
print(docs[1].metadata)
print(docs[0].page_content)
