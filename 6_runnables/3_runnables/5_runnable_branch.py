from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableBranch,
)

load_dotenv()

template1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)
template2 = PromptTemplate(
    template="Summarize the following text \n {text}", input_variables=["text"]
)

model = OpenAI()
parser = StrOutputParser()

report_chain = template1 | model | parser
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, template2 | model | parser),
    RunnablePassthrough(),
)
# chain = report_chain | branch_chain or
chain = RunnableSequence(report_chain, branch_chain)

result = chain.invoke({"topic": "Gen AI"})
print(result)
