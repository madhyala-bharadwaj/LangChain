from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

url = "https://python.langchain.com/api_reference/core/language_models.html"
urls = [
    "https://python.langchain.com/api_reference/core/language_models.html",
    "https://python.langchain.com/v0.1/docs/integrations/text_embedding/",
]

# loader = WebBaseLoader(url)
loader = WebBaseLoader(urls)
docs = loader.load()

# print(len(docs))
# print(docs[0].metadata)
# print(docs[0].page_content)

load_dotenv()
model = OpenAI()
template = PromptTemplate(
    template="Answer the following question \n {question} from the following text {text}",
    input_variables=["question", "text"],
)
parser = StrOutputParser()
chain = template | model | parser
print(
    chain.invoke(
        {
            "question": "What are the different language models that are discussed?",
            "text": docs[0].page_content,
        }
    )
)
