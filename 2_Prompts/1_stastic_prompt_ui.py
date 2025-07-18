from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-summarization",
)

model = ChatHuggingFace(llm=llm)

st.header("Deepseek Research Tool")
st.write("This tool uses the DeepSeek R1 model to generate text.")

user_input = st.text_input("Enter your prompt:")

if st.button("Summarize"):
    if user_input:
        result = model.invoke(user_input)
        st.write(result.content)
