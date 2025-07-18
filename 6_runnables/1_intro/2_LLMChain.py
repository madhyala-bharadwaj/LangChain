from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Create prompt template
template = PromptTemplate(
    template="Suggest a catchy blog title about {topic}.", input_variables=["topic"]
)

# Create an LLMChain
chain = LLMChain(llm=model, prompt=template)

# Run the chain with the input topic
topic = input("Enter a topic for the blog: ")
output = chain.run(topic=topic)
print("Generated blog title:", output)
