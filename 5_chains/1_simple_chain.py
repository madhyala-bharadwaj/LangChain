from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528", task="text2text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template = PromptTemplate(
    template="Generate 5 interesting facts about {topic}", input_variables=["topic"]
)

chain = template | model | parser

result = chain.invoke({"topic": "Cricket"})

print(result)

chain.get_graph().print_ascii()  # Visualize the chain
