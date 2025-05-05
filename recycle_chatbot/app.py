import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from dotenv import load_dotenv
from typing import List
import os

######################### 환경 변수 로드 #########################
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    st.error("GROQ_API_KEY 환경 변수가 설정되지 않았습니다.")
    st.stop()

#########################  페이지 설정 #########################
st.set_page_config(page_title="Recycle Chatbot", page_icon="♻️")
st.title("♻️ 서울시 구별 재활용 챗봇")

#########################  구 선택 #########################
gu_list = ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구",
           "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구",
           "마포구", "서대문구", "서초구", "성동구", "성북구",
           "송파구", "양천구", "영등포구", "용산구", "은평구",
           "종로구", "중구", "중랑구"]

with st.sidebar:
    selected_gu = st.selectbox("구를 선택하세요", gu_list)

######################### RAG 검색 함수 (실제 구현 필요) #########################

# PDF 문서 로드 및 분할
@st.cache_resource
def load_and_split_documents(selected_gu: str):
    loaders = []
    
    # 공통 PDF 파일 로드
    common_pdf_path = "./data/공통.pdf"
    common_loader = PyPDFLoader(common_pdf_path)
    loaders.append(common_loader)
    
    # 선택된 구의 PDF 파일 로드
    gu_pdf_path = f"./data/{selected_gu}.pdf"
    if os.path.exists(gu_pdf_path):  # 해당 구의 PDF가 있을 경우만 로드
        gu_loader = PyPDFLoader(gu_pdf_path)
        loaders.append(gu_loader)
    
    all_docs = []
    for loader in loaders:
        docs = loader.load()
        all_docs.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)
    return split_docs

# 임베딩 및 벡터 DB 생성
@st.cache_resource
def create_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(_docs, embeddings)


# Groq 기반 RAG 체인 생성
def get_rag_chain(vectorstore, groq_api_key):
    retriever = vectorstore.as_retriever()

    # Groq LLM 설정
    llm = ChatGroq(api_key=groq_api_key, model_name="gemma2-9b-it")  # 최신 모델로 수정 가능

    # 프롬프트 템플릿 정의
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        너는 서울시의 재활용 정책을 잘 아는 안내 챗봇이야.\n
        아래 문서를 참고하여 사용자의 질문에 대해 정확하고 친절하게 한국어로 답해줘.\n

        [문서 요약]
        {context}

        [질문]
        {question}

        [답변]
        """,
    )

    # RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

def retrieve_documents(query: str) -> List[str]:
    # 여기에 FAISS, Chroma 등에서 검색하여 유사한 정책 문서 반환
    # 예시:
    return [
        "강남구는 투명 페트병을 별도로 분리 배출해야 하며...",
        "플라스틱은 비닐과 분리하여 배출해야 합니다..."
    ]

#########################  챗봇 #########################
# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner("📄 문서 로딩 중..."):
    docs = load_and_split_documents(selected_gu)
    vectorstore = create_vectorstore(docs)
    qa_chain = get_rag_chain(vectorstore, GROQ_API_KEY)

# 채팅 입력
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 사용자 질문 입력
user_input = st.chat_input("재활용 방법을 질문하세요.")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = qa_chain.invoke({"query": user_input})
        st.markdown(response["result"])

    # 채팅 기록 저장
    st.session_state.chat_history.append((user_input, response["result"]))
