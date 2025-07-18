from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import FAISS

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

# Manually retrieve relevant documents
query = "What are the key takeaways from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Manually pass retrieved documents to the model
prompt = f"Based on the following documents, answer the question: {query}\n\n{retrieved_docs}"
result = model.predict(prompt)

print("Answer:", result)
