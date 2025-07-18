from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="Suggest a catchy blog title about {topic}.", input_variables=["topic"]
)

topic = input("Enter a topic for the blog: ")

prompt = template.format(topic=topic)

print("Generated blog title", model.predict(prompt))
