# Reflexion을 LangGraph로 : 실패하면 반성문을 쓰고 다시 하기
# 노드 3개 : actor(풀기) → evaluator(채점) → reflect(반성문) → 다시 actor

from langchain_openai import ChatOpenAI
import os, re, operator
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"
llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',    # lg05에서 잘 돌아간 모델 그대로
    temperature= 0.7,                # 시도마다 다른 문장이 나오게
    max_tokens = 1000,
    api_key= api_key,
    base_url= base_url,
)

######################### 0.과제 정의 ###########################
# 단어 첫 글자가 가, 나, 다, ... 하 순서인 문장 (14단어)
LETTERS = "가나다라마바사아자차카타파하"
TASK = (f"단어 {len(LETTERS)}개로 된 한국어 문장 하나를 써. "
        f"각 단어의 첫 글자는 순서대로 {', '.join(LETTERS)} 이어야 해. "
        "(예: 가방 나비 다리 ...) 문장만 출력해.")

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

######################## 1. State 정의 ########################
class State(TypedDict):
    attempt  : str                               # 이번 시도 답
    ok       : bool                              # 채점 결과
    feedback : str                               # 틀린 점
    trial    : int                               # 몇 번째 시도
    memory   : Annotated[list, operator.add]     # 반성문 저장소 (계속 쌓임)

######################## 2. Node 정의 ########################
def actor(state: State):
    """Actor : 과제 + 최근 반성문 3개를 읽고 시도"""
    prompt = TASK
    if state["memory"]:
        prompt += "\n\n지난 실패에서 얻은 교훈:\n" + \
                  "\n".join(f"- {m}" for m in state["memory"][-3:])
    attempt = llm.invoke(prompt).content.strip()
    print(f"\n[시도 {state['trial'] + 1}] {attempt}")
    return {"attempt": attempt, "trial": state["trial"] + 1}

def evaluator(state: State):
    """Evaluator : 파이썬으로 규칙 검사 (LLM 아님)"""
    words = [re.sub(r"[^\w]", "", w) for w in state["attempt"].split()]
    words = [w for w in words if w]
    bad = []
    if len(words) != len(LETTERS):
        bad.append(f"단어가 {len(words)}개 ({len(LETTERS)}개여야 함)")
    for i, (w, want) in enumerate(zip(words, LETTERS)):
        if w[0] != want:
            bad.append(f"{i+1}번째 단어 '{w}'는 '{want}'로 시작해야 함")
    feedback = " / ".join(bad[:4]) or "모두 통과"
    print(f"   채점 : {'성공' if not bad else '실패'} | {feedback}")
    return {"ok": not bad, "feedback": feedback}

def reflect(state: State):
    """Self-Reflection : 왜 틀렸는지 + 다음 계획을 글로"""
    note = llm.invoke(
        f"과제: {TASK}\n내 답: {state['attempt']}\n채점 결과: {state['feedback']}\n"
        "왜 틀렸는지, 다음에는 어떻게 고칠지 2문장으로 써.").content.strip()
    print(f"   반성문 : {note}")
    return {"memory": [note]}                    # 바뀌는 건 메모리뿐 (가중치 X)

def route(state: State):
    """성공했거나 5번 넘으면 끝, 아니면 반성하러"""
    if state["ok"] or state["trial"] >= 5:
        return END
    return "reflect"

#################### 3. Graph 구성, Node 추가 ####################
graph_builder = StateGraph(State)
graph_builder.add_node("actor", actor)
graph_builder.add_node("evaluator", evaluator)
graph_builder.add_node("reflect", reflect)

####################4. 엣지추가 #############################
graph_builder.add_edge(START, "actor")
graph_builder.add_edge("actor", "evaluator")
graph_builder.add_conditional_edges("evaluator", route)   # 성공 → END / 실패 → reflect
graph_builder.add_edge("reflect", "actor")                # 반성문 들고 다시 시도

############################## 5.컴파일 #################################
graph = graph_builder.compile()

############################ 6. 그래프실행 #############################
result = graph.invoke({"attempt": "", "ok": False, "feedback": "", "trial": 0, "memory": []})

print("\n=========== 결과 ===========")
print("최종 :", "성공" if result["ok"] else "실패", f"({result['trial']}번째 시도)")
print("쌓인 반성문 :", len(result["memory"]), "개")

# 그래프 그림 저장 (lg07 방식)
os.makedirs('./_save/image/', exist_ok=True)
with open('./_save/image/lg10_Reflexion.png', 'wb') as f:
    f.write(graph.get_graph().draw_mermaid_png())
