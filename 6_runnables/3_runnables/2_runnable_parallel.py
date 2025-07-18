from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

template1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a tweet about {topic}",
)
template2 = PromptTemplate(
    template="Generate a linkedin post about {topic}",
    input_variables=["topic"],
)

model = OpenAI()
parser = StrOutputParser()

chain = RunnableParallel(
    {
        "tweet": RunnableSequence(template1, model, parser),
        "linkedin": RunnableSequence(template2, model, parser),
    }
)

result = chain.invoke({"topic": "Gen AI"})
print(result)
print(result["tweet"])
print(result["linkedin"])
