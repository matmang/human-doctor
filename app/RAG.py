import os
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# True로 설정 시 HuggingFace BAAI/bge-m3 임베딩 사용 (2.7GB)
# False 설정 시 OpenAIEmbeddings 사용 (OPENAI_API_KEY 입력 필요. 과금)
USE_BGE_EMBEDDING = True

if not USE_BGE_EMBEDDING:
    os.environ["OPENAI_API_KEY"] = ""

LANGSERVE_ENDPOINT = "http://localhost:8000/llm/"

if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

RAG_PROMPT_TEMPLATE = """
From now on, as a doctor, you must identify symptoms and provide answers through Q&A with patients. 
Before answering, please express the diagnosis progress for the disease you estimate as a percentage at the top, and complete the diagnosis when it reaches 100%. 
Proceed with the diagnosis with confidence in your answers, and continue to ask questions and obtain information about the patient's symptoms until the diagnosis progress reaches 100%. 
First, start with a diagnosis progress of 0%. 
Ask one question at a time. 
Do not expect Human's answer in your turn. 
Do not repeat your answer.
그리고, 반드시  한국어로 답하세요.

Question:
{question}

Context:
{context}

Diagnosis pregress:

Answer:"""

st.set_page_config(page_title="Human Docter 팀 최종 과제", page_icon="💬")
st.title("Team Human Docter")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]

def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource(show_spinner="Embedding file...")
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

with st.sidebar:
    file = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt", "docx"],
    )

retriever = embed_file()

print_history()

if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        ollama = RemoteRunnable(LANGSERVE_ENDPOINT)
        chat_container = st.empty()
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
        answer = rag_chain.stream(user_input)
        chunks = []
        for chunk in answer:
            chunks.append(chunk)
            chat_container.markdown("".join(chunks))
        add_history("ai", "".join(chunks))
