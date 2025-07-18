from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful {domain} assistant"),
        ("human", "Explain in simple terms, about {topic}"),
    ]
)

prompt = chat_prompt.invoke({"domain": "cricket", "topic": "batting techniques"})

print(prompt)
