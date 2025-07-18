from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

llm = OpenAI()
template1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a joke about {topic}",
)
template2 = PromptTemplate(
    input_variables=["joke"], template="Now, tell me why this joke is funny: {joke}"
)
parser = StrOutputParser()

chain = RunnableSequence(template1, llm, parser, template2, llm, parser)

print(chain.invoke({"topic": "Gen AI"}))
