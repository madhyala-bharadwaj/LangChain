from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
)

load_dotenv()

template1 = PromptTemplate(
    template="Generate a joke about {topic}", input_variables=["topic"]
)
template2 = PromptTemplate(
    template="Explain the following joke \n {joke}", input_variables=["joke"]
)

model = OpenAI()
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(template1, model, parser)
parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(template2, model, parser),
    }
)
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic": "Gen AI"})

print(result)
print(result["joke"])
print(result["explanation"])
