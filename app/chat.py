import os
import glob
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.prompts.few_shot import (FewShotChatMessagePromptTemplate, FewShotPromptTemplate)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain_core.runnables import RunnablePassthrough

# LLM 모델 초기화
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            From now on, as a doctor, you must identify symptoms and provide answers through Q&A with patients. 
            Before answering, please express the diagnosis progress for the disease you estimate as a percentage at the top, and complete the diagnosis when it reaches 100%. 
            Proceed with the diagnosis with confidence in your answers, and continue to ask questions and obtain information about the patient's symptoms until the diagnosis progress reaches 100%. 
            First, start with a diagnosis progress of 0%. 
            Ask one question at a time. 
            Do not expect Human's answer in your turn. 
            Do not repeat your answer.
            

            Example answer:
            
            Diagnosis progress: 20%
            
            Doesn’t it cause pain when you press and release your lower right abdomen?
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50,
#     separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
#     length_function=len
# )

# loader = UnstructuredFileLoader("./.cache/files/naver_medical.pdf")

# docs = loader.load_and_split(text_splitter=text_splitter)

# embeddings = HuggingFaceBgeEmbeddings(
#     model_name= "BAAI/bge-m3",
#     model_kwargs = {
#         "device": "cuda"
#     },
#     encode_kwargs = {
#         "normalize_embeddings": True
#     }
# )

# cache_dir = LocalFileStore("./.cache/embeddings/naver_medical.pdf")

# cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir) # 미리 처리해놓은 벡터 DB에서 검색해오기

# vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# 최종 체인 정의
chain = prompt | llm.bind(stop=["Human:"]) | StrOutputParser()