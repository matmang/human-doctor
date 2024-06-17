import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

USE_BGE_EMBEDDING = True  # 사용할 임베딩 모델을 설정

def embed_file(file_path):
    # 파일 내용 읽기
    with open(file_path, "rb") as file:
        file_content = file.read()

    # 파일 저장 경로 설정
    cache_file_path = f"./.cache/files/{os.path.basename(file_path)}"
    with open(cache_file_path, "wb") as f:
        f.write(file_content)

    # 캐시 디렉토리 설정
    cache_dir = LocalFileStore(f"./.cache/embeddings/{os.path.basename(file_path)}")

    # 텍스트 분할 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
        length_function=len
    )

    # 문서 로더 설정
    loader = UnstructuredFileLoader(cache_file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)

    # 임베딩 설정
    if USE_BGE_EMBEDDING:
        model_name = "BAAI/bge-m3"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    else:
        embeddings = OpenAIEmbeddings()

    # 캐시된 임베딩 설정
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    FAISS.from_documents(docs, embedding=cached_embeddings)
    print(f"Finished processing and caching file: {file_path}")

def process_all_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            print(f"Processing file: {filename}")
            embed_file(file_path)

# 지정된 디렉터리의 모든 파일 처리
process_all_files_in_directory("./.cache/files")