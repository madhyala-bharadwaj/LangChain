from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

loader = TextLoader("7_document_loaders\cricket.txt", encoding="utf-8")
docs = loader.load()

print(docs)
print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

load_dotenv()
model = OpenAI()
template = PromptTemplate(
    template="Write a summary for the following poem \n {poem}",
    input_variables=["poem"],
)
parser = StrOutputParser()

chain = template | model | parser
print(chain.invoke({"poem": docs[0].page_content}))
