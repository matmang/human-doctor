from langchain_community.chat_models import ChatOllama
from langchain.chat_models.openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
import openai

# os.environ['OPENAI_API_KEY'] = ''

# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")
# llm = ChatOpenAI(temperature=0, max_tokens=2048, model="gpt-3.5-turbo")