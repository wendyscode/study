
# 1주차 논문 4편을 검색해 답하는 RAG 챗봇

import os
import re
import json
import copy
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# ==========================================
# 00. 환경 설정
# ==========================================

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"


# ==========================================
# 01. 데이터를 불러온다. 지정한 폴더의 PDF를 모두 읽는다.
# ==========================================

path = Path(r"C:\Users\Admin\Desktop\논문\1\1주차_원본\1주차")

pdf_files = sorted(path.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {path}")

pdf_docs = []

for pdf_file in pdf_files:

    docs = PyPDFLoader(str(pdf_file)).load()

    for doc in docs:
        doc.metadata["paper_name"] = pdf_file.name
        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1

    pdf_docs.extend(docs)

print("논문 PDF 개수 :", len(pdf_files))
print("원본 페이지 개수 :", len(pdf_docs))


# ==========================================
# 02. 데이터를 자른다. - 청킹
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(pdf_docs)

print("청킹 후 문서 개수 :", len(split_docs))


# ==========================================
# 03. 임베딩
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}
)


# 벡터 DB
DB_PATH = "./_db/1주차논문_bge/"

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name="week1_papers",
)

if vector_store._collection.count() == 0:
    vector_store.add_documents(split_docs)

print("벡터 DB 문서 개수 :", vector_store._collection.count())


# ==========================================
# 04. Retriever 리트리버
# ==========================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# ==========================================
# 05. 모델 연결
# ==========================================

model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_tokens=1800,
    api_key=api_key,
    base_url=base_url,
)


# ==========================================
# 06. RAG 체인 만들기
# ==========================================

prompt = ChatPromptTemplate.from_template("""
너는 1주차 객체 탐지 논문 4편을 가르치는 학습 챗봇이다.

논문:
- R-CNN: Rich feature hierarchies for accurate object detection and semantic segmentation
- Fast R-CNN
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- Mask R-CNN

사용자의 질문에 대해 검색된 논문 컨텍스트를 바탕으로 답변해라.

규칙:
1. 논문 내용을 근거로 답변한다.
2. 초보자도 이해할 수 있도록 쉽게 설명한다.
3. 논문에 없는 내용을 사실처럼 만들어내지 않는다.
4. 가능하면 논문명과 페이지를 표시한다.

논문 컨텍스트:
{context}

질문:
{input}

답변:
""")

# 검색된 문서의 논문명과 페이지를 컨텍스트에 함께 넣는다.
document_prompt = PromptTemplate.from_template(
    "[출처: {paper_name}, p. {page_number}]\n{page_content}"
)

docu_chain = create_stuff_documents_chain(
    model,
    prompt,
    document_prompt=document_prompt,
)

rag_chain = create_retrieval_chain(
    retriever,
    docu_chain
)


# ==========================================
# 07. 퀴즈 논문 설정
# ==========================================

PAPERS = [
    "R-CNN Rich feature hierarchies for accurate object detection and semantic segmentation",
    "Fast R-CNN",
    "Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks",
    "Mask R-CNN"
]


# ==========================================
# 08. 객관식 문제 생성 함수
# ==========================================

def generate_question(question_number, previous_questions):

    # 논문별 5문제씩 출제한다.
    paper_index = min((question_number - 1) // 5, 3)

    paper_name = PAPERS[paper_index]

    # 해당 논문의 내용을 검색한다.
    docs = retriever.invoke(paper_name)

    context = "\n\n".join([
        f"[출처: {doc.metadata.get('paper_name', '')}, "
        f"p. {doc.metadata.get('page_number', '')}]\n"
        f"{doc.page_content}"
        for doc in docs
    ])

    previous_text = "\n".join(previous_questions)

    quiz_prompt = f"""
너는 객체 탐지 논문 학습을 위한 객관식 문제 출제자다.

대상 논문:
{paper_name}

논문 내용:
{context}

이전 출제 문제:
{previous_text if previous_text else "아직 출제된 문제가 없음"}

위 논문 내용을 바탕으로 객관식 문제를 정확히 1개 출제해라.

[출제 규칙]

1. 보기 A, B, C, D 총 4개를 제공한다.
2. 정답은 반드시 하나만 존재해야 한다.
3. 반드시 논문 내용에 근거해서 출제한다.
4. 초보자도 이해할 수 있는 문제를 출제한다.
5. 이전 문제와 최대한 중복되지 않도록 한다.
6. 오답 보기에도 논문과 관련된 개념을 사용한다.
7. 정답과 해설을 정확하게 작성한다.
8. 문제는 한국어로 작성한다.

[출력 형식]

반드시 JSON 형식으로만 출력한다.

{{
    "question": "문제 내용",
    "A": "보기 A",
    "B": "보기 B",
    "C": "보기 C",
    "D": "보기 D",
    "answer": "A",
    "explanation": "정답인 이유에 대한 쉬운 해설"
}}
"""

    result = model.invoke(quiz_prompt)

    raw = result.content

    if not isinstance(raw, str):
        raw = str(raw)

    raw = raw.strip()

    # JSON 추출
    match = re.search(r"\{.*\}", raw, re.DOTALL)

    if not match:
        raise ValueError("문제 생성 결과가 올바른 JSON 형식이 아닙니다.")

    data = json.loads(match.group())

    # 문제 데이터 검증
    required_keys = [
        "question",
        "A",
        "B",
        "C",
        "D",
        "answer",
        "explanation"
    ]

    for key in required_keys:

        if not data.get(key):
            raise ValueError(f"문제 데이터 누락: {key}")

    if data["answer"] not in ["A", "B", "C", "D"]:
        raise ValueError("정답은 A, B, C, D 중 하나여야 합니다.")

    return data


# ==========================================
# 09. 문제 출력 함수
# ==========================================

def format_question(data, number):

    return f"""
📚 질문 {number}

{data['question']}

A) {data['A']}

B) {data['B']}

C) {data['C']}

D) {data['D']}

정답을 A, B, C, D 중 하나로 입력하세요.
"""


# ==========================================
# 10. 퀴즈 상태 관리
# ==========================================

initial_state = {
    "mode": "idle",
    "question": None,
    "correct_answer": None,
    "explanation": None,
    "score": 0,
    "total": 0,
    "previous_questions": []
}


# ==========================================
# 11. Gradio 챗봇 답변 함수
# ==========================================

def answer_invoke(message, history, state):

    message = message.strip()

    state = copy.deepcopy(state or initial_state)

    history = history or []

    # --------------------------------------
    # 1. 퀴즈 시작
    # --------------------------------------

    if message.upper() in ["GO", "시작", "퀴즈 시작"]:

        state = copy.deepcopy(initial_state)

        try:

            data = generate_question(
                1,
                state["previous_questions"]
            )

            state["question"] = data
            state["correct_answer"] = data["answer"]
            state["explanation"] = data["explanation"]

            state["previous_questions"].append(
                data["question"]
            )

            state["mode"] = "waiting"

            response = (
                "🎯 1주차 논문 퀴즈를 시작합니다!\n"
                + format_question(data, 1)
            )

        except Exception as e:

            state["mode"] = "idle"

            response = f"문제 생성 오류가 발생했습니다.\n{e}"

    # --------------------------------------
    # 2. 사용자 답변 채점
    # --------------------------------------

    elif state["mode"] == "waiting":

        answer = message.strip().upper()

        # A, B, C, D 이외의 입력 처리
        if answer not in ["A", "B", "C", "D"]:

            response = (
                "⚠️ A, B, C, D 중 하나로 답변해 주세요.\n\n"
                "현재 문제에 먼저 답변해 주세요."
            )

            history.append({
                "role": "user",
                "content": message
            })

            history.append({
                "role": "assistant",
                "content": response
            })

            return "", history, state

        # Python에서 직접 채점한다.
        correct = state["correct_answer"]

        state["total"] += 1

        if answer == correct:

            state["score"] += 1

            feedback = "🎉 정답입니다!"

        else:

            feedback = f"❌ 오답입니다! 정답은 {correct}입니다."

        # 채점 결과와 해설
        response = f"""
{feedback}

정답: {correct}

📖 해설:
{state['explanation']}

📊 현재 점수:
{state['score']}문제 정답 / 총 {state['total']}문제
"""

        # ----------------------------------
        # 3. 다음 문제 자동 출제
        # ----------------------------------

        next_number = state["total"] + 1

        if next_number <= 20:

            try:

                data = generate_question(
                    next_number,
                    state["previous_questions"]
                )

                state["question"] = data
                state["correct_answer"] = data["answer"]
                state["explanation"] = data["explanation"]

                state["previous_questions"].append(
                    data["question"]
                )

                response += (
                    "\n\n"
                    + format_question(data, next_number)
                )

                state["mode"] = "waiting"

            except Exception as e:

                state["mode"] = "error"

                response += (
                    f"\n\n다음 문제 생성 중 오류가 발생했습니다.\n{e}"
                )

        else:

            state["mode"] = "finished"

            response += """

🏆 축하합니다!

1주차 논문 20문제를 모두 완료했습니다.

최종 점수를 확인해 보세요!
"""

    # --------------------------------------
    # 4. 일반 논문 질문
    # --------------------------------------

    elif state["mode"] in ["idle", "finished"]:

        try:

            result = rag_chain.invoke({
                "input": message
            })

            response = result["answer"]

        except Exception as e:

            response = f"답변 생성 오류가 발생했습니다.\n{e}"

    # --------------------------------------
    # 5. 오류 상태 처리
    # --------------------------------------

    elif state["mode"] == "error":

        response = (
            "퀴즈 진행 중 오류가 발생했습니다.\n\n"
            "GO를 입력하면 퀴즈를 다시 시작할 수 있습니다."
        )

    # --------------------------------------
    # 6. 대화 기록 저장
    # --------------------------------------

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": response
    })

    return "", history, state


# ==========================================
# 12. Gradio 챗봇
# ==========================================

import gradio as gr

with gr.Blocks() as demo:

    gr.Markdown("# 📚 1주차 논문 학습 챗봇")

    # 퀴즈 상태 저장
    quiz_state = gr.State(
        copy.deepcopy(initial_state)
    )

    # 채팅창
    # Gradio 6.29.1에서는 type 인자를 사용하지 않는다.
    chatbot = gr.Chatbot(
        value=[
            {
                "role": "assistant",
                "content": """안녕하세요! 😊

1주차 논문 학습 챗봇입니다.

R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN 논문 4편을 바탕으로 답변해요.

쉬운 설명부터 발표에 필요한 자세한 내용까지 질문해보세요.

예시:

- 네 논문의 핵심 아이디어와 발전 흐름을 알려주세요.
- Faster R-CNN의 RPN을 쉽게 설명해주세요.
- Fast R-CNN과 Mask R-CNN의 차이를 비교해주세요.
- Mask R-CNN을 발표할 수 있게 정리해주세요.
- 논문 내용을 퀴즈로 내주세요.

퀴즈를 시작하려면 GO를 입력하세요!

진행 방식:
문제 1개 출제 → 답변 → 즉시 채점 → 해설 → 다음 문제"""
            }
        ],
        height=550
    )

    # 사용자 입력창
    msg = gr.Textbox(
        placeholder="GO를 입력하거나 논문에 대해 질문하세요.",
        label="질문 및 답변"
    )

    # 메시지 제출
    msg.submit(
        answer_invoke,
        inputs=[
            msg,
            chatbot,
            quiz_state
        ],
        outputs=[
            msg,
            chatbot,
            quiz_state
        ]
    )


# ==========================================
# 13. 실행
# ==========================================

demo.launch()
