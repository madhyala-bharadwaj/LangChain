from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)

# llm.invoke("What is the capital of France?").then(
#     lambda response: print(f"Response: {response}")
# ).catch(lambda error: print(f"Error: {error}"))

response = llm.invoke("What is the capital of France?")
print(f"Response: {response}")  # string output
