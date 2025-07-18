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

template1 = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Generate a 5 line summary from the following text \n {report}",
    input_variables=["report"],
)

chain = template1 | model | parser | template2 | model | parser

chain.get_graph().print_ascii()

result = chain.invoke({"topic": "Generative AI"})

print(result)
