from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

documents = [
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Kolkata is known for its rich cultural heritage.",
    "Chennai is famous for its classical music and dance.",
    "Bangalore is known as the Silicon Valley of India.",
]

result = embeddings.embed_documents(documents)

print(f"Embeddings: {result}")
