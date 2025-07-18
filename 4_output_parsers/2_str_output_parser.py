from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    template="Write a 5 line summary on the following text: \n {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "black hole"})

print(result)
