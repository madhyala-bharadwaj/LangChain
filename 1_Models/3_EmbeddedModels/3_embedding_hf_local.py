from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Hugging Face is a popular platform for NLP models."
# text = [
#     "Hugging Face is a popular platform for NLP models.",
#     "It provides a wide range of pre-trained models for various tasks.",
#     "The platform supports both PyTorch and TensorFlow.",
#     "Hugging Face also offers an easy-to-use API for model deployment.",
#     "The community around Hugging Face is very active and supportive."
# ]

vector = embedding.embed_query(text)
# embedding.embed_documents(text)

print(f"Vector: {str(vector)}")
