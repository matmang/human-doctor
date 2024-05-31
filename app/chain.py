from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="EEVE-Korean-10.8B:latest")

prompt = ChatPromptTemplate.from_template("{topic} 에 대하여 간략히 설명해 줘.")

chain = prompt | llm | StrOutputParser()