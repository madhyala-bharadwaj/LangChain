from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528", task="text2text-generation"
)

model = ChatHuggingFace(llm=llm)

# Detailed report prompt
template1 = PromptTemplate(
    template="Write a detailed report on the {topic}", input_variables=["topic"]
)

# Summary prompt
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text: /n {text}",
    input_variables=["text"],
)

prompt1 = template1.invoke({"topic": "black hole"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result1.content})
result = model.invoke(prompt2)

print(result.content)
