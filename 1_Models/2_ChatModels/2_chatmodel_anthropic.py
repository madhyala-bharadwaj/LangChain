from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-2", temperature=0.8, max_completion_tokens=50)

result = model.invoke("Write a 5 line poem on cricket")

print(f"Response: {result.content}")
