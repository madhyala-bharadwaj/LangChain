from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ["HF_HOME"] = "D:/HuggingFace_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"temperature": 1.0, "max_new_tokens": 100},
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Write a 5 line poem on cricket")

print(f"Response: {result.content}")
