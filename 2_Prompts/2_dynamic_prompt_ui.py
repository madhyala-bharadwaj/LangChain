from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-summarization",
)

model = ChatHuggingFace(llm=llm)

st.header("Deepseek Research Tool")

paper_title = st.text_input("Enter the title of the research paper:")

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Length of Explanation",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (Detailed Explanation)",
    ],
)

# template = PromptTemplate(
#     input_variables=["paper_title", "style_input", "length_input"],
#     template="""
#         Please summarize the research paper titled "{paper_title}" with the following specifications:
#         Explanation Style: {style_input}
#         Explanation Length: {length_input}
#         1. Mathematical Details:
#             - Include relevant mathematical equations if present in the paper.
#             - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
#         2. Analogies:
#             - Use relatable analogies to simplify complex ideas.
#         If certain information is not available in the paper, respond with: \"Insufficient information available\" instead of guessing.
#         Ensure the summary is clear, accurate, and aligned with the provided style and length.
#         """,
#     validate_template=True,
# )

template = load_prompt("2_Prompts\\template.json")

# prompt = template.invoke(
#     {
#         "paper_title": paper_title,
#         "style_input": style_input,
#         "length_input": length_input,
#     }
# )

# if st.button("Explain"):
#     result = model.invoke(prompt)
#     st.write(result.content)


# using LangChain's chaining mechanism
if st.button("Explain"):
    chain = template | model
    result = chain.invoke(
        {
            "paper_title": paper_title,
            "style_input": style_input,
            "length_input": length_input,
        }  # input to template
    )
    st.write(result.content)
