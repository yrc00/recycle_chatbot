import os
from dotenv import load_dotenv

import streamlit as st
from typing import List
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

######################### 환경 변수 로드 #########################

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    st.error("GROQ_API_KEY 환경 변수가 설정되지 않았습니다.")
    st.stop()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if COHERE_API_KEY is None:
    st.error("COHERE_API_KEY 환경 변수가 설정되지 않았습니다.")
    st.stop()

#########################  페이지 설정 #########################

st.set_page_config(page_title="Recycle Chatbot", page_icon="♻️")
st.title("♻️ 서울시 구별 재활용 챗봇")

#########################  구 선택 #########################
gu_list = ["용산구"]

with st.sidebar:
    selected_gu = st.selectbox("구를 선택하세요", gu_list)

######################### RAG 검색 함수 (실제 구현 필요) #########################

# PDF 문서 로드 및 분할
@st.cache_resource
def load_and_split_documents(selected_gu: str):
    loaders = []
    
    # 공통 PDF 파일 로드
    if os.path.exists("./data/공통.pdf"):
        common_pdf_path = "./data/공통.pdf"
        common_loader = PyPDFLoader(common_pdf_path)
        loaders.append(common_loader)
    else:
        st.warning("공통 PDF 파일이 없습니다. 구별 PDF만 사용됩니다.")
        common_loader = None
    
    # 선택된 구의 PDF 파일 로드
    gu_pdf_path = os.path.join(os.path.dirname(__file__), f"data/{selected_gu}.pdf")
    if os.path.exists(gu_pdf_path):  # 해당 구의 PDF가 있을 경우만 로드
        gu_loader = PyPDFLoader(gu_pdf_path)
        loaders.append(gu_loader)
    else:
        st.warning(f"{selected_gu}에 대한 PDF 파일이 없습니다. 공통 PDF만 사용됩니다.")
        gu_loader = None
        loaders.append(common_loader)
    
    all_docs = []
    for loader in loaders:
        docs = loader.load()
        all_docs.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ""],
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = splitter.split_documents(all_docs)
    return split_docs

# 문서 내용을 정리(clean)하는 함수
def clean_documents(documents: List[Document]) -> str:
    cleaned = [
        doc.page_content.replace("\n", " ").strip()
        for doc in documents
    ]
    return "\n\n".join(cleaned)

# 전체 체인 생성 함수
def get_rag_chain(split_documents, groq_api_key: str, cohere_api_key: str) -> tuple[Runnable, ContextualCompressionRetriever]:
    # 1. 기본 retriever
    base_retriever = FAISS.from_documents(
        split_documents,
        CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=cohere_api_key)
    ).as_retriever(search_kwargs={"k": 10})

    # 2. Cohere reranker 기반 compression retriever
    compressor = CohereRerank(
        model="rerank-multilingual-v3.0", 
        top_n=2, 
        cohere_api_key=cohere_api_key
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 3. LLM (Groq)
    llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

    # 4. 프롬프트 템플릿
    system_prompt = (
        "You are a helper who tells the user how to dispose of garbage. "
        "Given the context, answer in Korean. "
        "If you don't know the answer, say you don't know. "
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 5. 문서 결합 체인
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # 6. 전체 retrieval chain
    rag_chain = create_retrieval_chain(compression_retriever, combine_docs_chain)

    return rag_chain, compression_retriever

#########################  챗봇 #########################

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner("📄 문서 로딩 중..."):
    docs = load_and_split_documents(selected_gu)
    qa_chain, compression_retriever = get_rag_chain(docs, GROQ_API_KEY, COHERE_API_KEY)


# 채팅 히스토리 세션
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# 이전 대화 표시
for msg, resp in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg)
    with st.chat_message("assistant"):
        st.markdown(resp)

# 사용자 질문 입력
user_input = st.chat_input("재활용 방법을 질문해보세요!")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            # 문서 압축 수행
            compressed_docs = compression_retriever.invoke(user_input)
            context = clean_documents(compressed_docs)

            # 체인 실행
            result = qa_chain.invoke({
                "input": user_input,
                "context": context
            })
            st.markdown(result["answer"])

            # 대화 기록 저장
            st.session_state.chat_history.append((user_input, result["answer"]))
