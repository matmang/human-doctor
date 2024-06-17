import os
import numpy as np
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics import ndcg_score
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# 설정
USE_BGE_EMBEDDING = True

LANGSERVE_ENDPOINT = "http://localhost:8000/llm/"

if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

RAG_PROMPT_TEMPLATE = """
From now on, as a doctor, you must identify symptoms and provide answers through Q&A with patients.
Please describe the suspected diagnosis as kindly and in detail as possible, and explain treatment options.
When diagnosing, refer to the context provided and ensure each question is relevant to the symptoms described.
The diagnosis must be clearly stated.
그리고, 반드시  한국어로 답하세요.

Question:
{question}

Context:
{context}

Answer:"""

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

def embed_file():
    file_path = f"./.cache/files/naver_medical.pdf"

    cache_dir = LocalFileStore(f"./.cache/embeddings/naver_medical.pdf")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)

    if USE_BGE_EMBEDDING:
        model_name = "BAAI/bge-m3"
        model_kwargs = {
            "device": "cuda"
        }
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    else:
        embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1, 'fetch_k': 50, 'lambda_mult': 0.5}, search_type='mmr')
    return retriever

def evaluate_qa(dataset, retriever):
    ollama = RemoteRunnable(LANGSERVE_ENDPOINT)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | ollama.bind(stop=["Human:", "사용자:", "질문:"])
        | StrOutputParser()
    )

    results = []

    for item in tqdm(dataset, desc="evaluation"):
        instruction = item["instruction"]
        true_output = item["output"]
        
        generated_answer = "".join([chunk for chunk in rag_chain.stream(instruction)])
        
        results.append({
            "question": instruction,
            "output": true_output,
            "label": generated_answer
        })
    
    return results

def calculate_ndcg(y_true, y_pred):
    relevance_scores = []

    for true, pred in zip(y_true, y_pred):
        if true[0] in pred[0]:
            relevance_scores.append([1])
        else:
            relevance_scores.append([0])

    return ndcg_score(y_true, relevance_scores)

# 데이터셋 로드
dataset = load_dataset("mncai/MedGPT-5k-ko")["train"].select(range(100))

# 임베딩 파일 준비
retriever = embed_file()

# QA 평가
results = evaluate_qa(dataset, retriever)

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)

# CSV 파일로 저장
results_df.to_csv("qa_results.csv", index=False)

print("QA 평가 결과가 'qa_results.csv' 파일에 저장되었습니다.")