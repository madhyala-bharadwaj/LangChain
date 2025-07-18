from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

# Load the document
loader = TextLoader("docs.txt")
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings nd store in FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Create a retriever (fetches relevant chunks based on a query)
retriever = vectorstore.as_retriever()

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Create a RetrievalQA chain
chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

query = "What are the key takeaways from the document?"
# retrieved_docs = retriever.get_relevant_documents(query)
# prompt = f"Based on the following documents, answer the question: {query}\n\n{retrieved_docs}"
# result = model.predict(prompt)

result = chain.run(query)

print("Answer:", result)
