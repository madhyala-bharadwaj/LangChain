from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()


class SentimentOutput(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=SentimentOutput)

template1 = PromptTemplate(
    template="Classify the sentiment of following feedback text as positive or negative \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

classifier_chain = template1 | model | parser2

# result = classifier_chain.invoke({"feedback": "I love the new features of this product!"}).sentiment
# print(result)

template2 = PromptTemplate(
    template="Write an appropriate response to the positive feedback \n {feedback}",
    input_variables=["feedback"],
)
template3 = PromptTemplate(
    template="Write an appropriate response to the negative feedback \n {feedback}",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", template2 | model | parser),
    (lambda x: x.sentiment == "negative", template3 | model | parser),
    default=RunnableLambda(
        lambda x: "Couldn't classify the sentiment of the feedback."
    ),
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "I love the new features of this product!"})
print(result)

chain.get_graph().print_ascii()
