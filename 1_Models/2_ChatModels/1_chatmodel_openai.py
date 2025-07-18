from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.8, max_completion_tokens=50)

result = model.invoke("Write a 5 line poem on cricket")

print(f"Response: {result}")

"""
Not a plain text string output, but a ChatMessage object including meta data.
Can access the content with result.content
"""

print(f"Response content: {result.content}")  # string output
