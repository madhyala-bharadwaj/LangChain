from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="7_document_loaders/pdfs", glob="*.pdf", loader_cls=PyPDFLoader
)

# docs = loader.load()

# print(len(docs))
# print(docs[0].page_content)
# print(docs[9].metadata)
# for doc in docs:
# print(doc.metadata)


docs = loader.lazy_load()
for doc in docs:
    print(doc.metadata)
