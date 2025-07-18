from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)

# def greet(name):
#     return f"Hello {name}"
# print(RunnableLambda(greet).invoke("Bharat"))

load_dotenv()

template = PromptTemplate(template="Give a joke on {topic}", input_variables=["topic"])
model = OpenAI()
parser = StrOutputParser()


def count_words(text):
    return len(text.split())


joke_gen_chain = RunnableSequence(template, model, parser)
parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(
            count_words
        ),  # RunnableLambda(lambda x: len(x.split()))
    }
)
chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = chain.invoke({"topic": "Gen AI"})

print(result)
print(result["joke"])
print(result["word_count"])
