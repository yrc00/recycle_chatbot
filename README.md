# Recycle Chatbot

서울시 재활용 챗봇입니다. 

구 별로 재활용 정책을 반영하여 답변합니다. 

**analysis**: 재활용 분석 데이터
**recycle_chatbot**: 서울시 재활용 챗봇

---

## Local 사용법

```
git clone https://github.com/yrc00/recycle_chatbot.git

cd recycle_chatbot
```
- git clone을 사용하여 코드 클론


```
# 새로운 가상환경
conda create -n recycle

# 패키지 설치
pip install -r requirements.txt
```
- 새로운 가상 환경을 생성한 뒤 실행에 필요한 패키지 설치


```
# GROQ API 설정
GROQ_API_KEY = "your_groq_api"
```
- .env_example 파일의 이름을 .env로 수정
- your_groq_api 자리에 groq api 작성 후 저장