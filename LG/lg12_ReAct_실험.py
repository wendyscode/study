# lg05_ToolNode.py 확장판 : 논문 ReAct를 따라 해 보는 실험
# ① Thought를 눈에 보이게  ② tools 노드 직접 구현(검색어 기록)
# ③ 같은 검색 반복 / 검색 횟수 초과 시 멈추기 (논문 한계 : 반복 루프)
# ④ HotpotQA 10문제 : 검색 없이(CoT) vs ReAct 비교

from langchain_openai import ChatOpenAI
import os, re, json, operator, requests
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"
llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature= 0,
    max_tokens = 4000,               # 속으로 생각하는 토큰까지 넉넉하게
    api_key= api_key,
    base_url= base_url,
)

######################### 0.Tool 정의 ###########################
from langchain_tavily import TavilySearch
web_search = TavilySearch(max_results=3)
tools = [web_search]
llm_with_tools = llm.bind_tools(tools)

# 논문처럼 Thought를 먼저 쓰게 하는 지시문 ①
SYSTEM = ("You answer multi-hop questions by searching the web step by step.\n"
          "Before every search, write ONE line starting with 'Thought:' that says what you need next.\n"
          "When you know the answer, reply with 'Final Answer: <short answer, a few words>'.")
MAX_SEARCH = 6      # 검색은 최대 6번까지 ③

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

######################## 1. State 정의 ########################
class State(TypedDict):
    messages    : Annotated[list, add_messages]      # 대화 기록 (lg05와 같음)
    searches    : Annotated[list, operator.add]      # 지금까지 한 검색어 ②
    stop_reason : str                                # 왜 멈췄는지 ③

######################## 2. Node 정의 ########################
def chatbot(state: State):
    """Thought + Action : 생각을 한 줄 쓰고 무엇을 검색할지 정함"""
    msg = llm_with_tools.invoke([SystemMessage(SYSTEM)] + state["messages"])
    return {"messages": [msg]}

def tools_node(state: State):
    """Observation : 검색을 실제로 실행하고 검색어를 기록 (ToolNode 직접 구현) ②"""
    out, queries = [], []
    for tc in state["messages"][-1].tool_calls:
        result = web_search.invoke(tc["args"])
        if isinstance(result, dict):
            text = " / ".join(r.get("content", "")[:200] for r in result.get("results", [])[:3])
        else:
            text = str(result)[:600]
        out.append(ToolMessage(content=text or "no result", tool_call_id=tc["id"]))
        queries.append(tc["args"].get("query", ""))
    return {"messages": out, "searches": queries}

def stop_and_answer(state: State, reason: str):
    """검색을 막고, 지금까지 찾은 정보로 답하게 함 ③"""
    blocked = [ToolMessage(content=f"Search blocked ({reason}). Answer with what you have.",
                           tool_call_id=tc["id"]) for tc in state["messages"][-1].tool_calls]
    msg = llm.invoke([SystemMessage(SYSTEM)] + state["messages"] + blocked +
                     [HumanMessage("Stop searching. Give the Final Answer now.")])
    return {"messages": blocked + [msg], "stop_reason": reason}

def stop_repeat(state: State):  return stop_and_answer(state, "같은 검색 반복")
def stop_limit(state: State):   return stop_and_answer(state, "검색 횟수 초과")

def norm(s):
    return re.sub(r"[^a-z0-9가-힣 ]", "", str(s).lower()).strip()

def route(state: State):
    """tools_condition을 직접 확장 : 끝 / 검색 / 반복 차단 / 횟수 차단"""
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END                                            # 검색할 게 없으면 Finish
    new = [norm(tc["args"].get("query", "")) for tc in last.tool_calls]
    done = [norm(q) for q in state["searches"]]
    if any(q in done for q in new):
        return "stop_repeat"                                  # 같은 검색어 또 나오면 차단
    if len(state["searches"]) + len(new) > MAX_SEARCH:
        return "stop_limit"                                   # 검색 6번 넘으면 차단
    return "tools"

#################### 3. Graph 구성, Node 추가 ####################
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools_node)
graph_builder.add_node("stop_repeat", stop_repeat)
graph_builder.add_node("stop_limit", stop_limit)

####################4. 엣지추가 #############################
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", route)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("stop_repeat", END)
graph_builder.add_edge("stop_limit", END)

############################## 5.컴파일 #################################
graph = graph_builder.compile()

os.makedirs('./_save/image/', exist_ok=True)
try:
    with open('./_save/image/lg12_ReAct.png', 'wb') as f:
        f.write(graph.get_graph().draw_mermaid_png())
except Exception as e:
    print("그래프 그림 저장 실패 (실험은 계속) :", e)

############################ 6. 실험 함수 #############################
def final_answer(text):
    text = str(text)
    return text.split("Final Answer:")[-1].strip() if "Final Answer:" in text else text.strip()

def correct(pred, gold):
    p, g = norm(pred), norm(gold)
    return bool(g) and (p == g or g in p)       # 느슨한 채점 : 정답이 답 안에 들어 있으면 정답

def run_cot(q):
    """비교군 : 검색 없이 머릿속 지식으로만 (CoT)"""
    out = llm.invoke(f"Question: {q}\nThink step by step, then reply 'Final Answer: <short answer>'.").content
    return final_answer(out)

def run_react(q):
    res = graph.invoke({"messages": [("user", q)], "searches": [], "stop_reason": ""},
                       config={"recursion_limit": 30})
    return final_answer(res["messages"][-1].content), res

def show_trace(res):
    """논문 형식으로 출력 : Thought → Action → Observation → Finish"""
    for msg in res["messages"]:
        if msg.type == "human":
            print("Question    :", str(msg.content)[:100])
        elif msg.type == "ai" and getattr(msg, "tool_calls", None):
            th = str(msg.content).replace("Thought:", "").strip()
            print("Thought     :", th[:100] if th else "(생략)")
            for tc in msg.tool_calls:
                print("Action      : Search[" + tc["args"].get("query", "") + "]")
        elif msg.type == "tool":
            print("Observation :", str(msg.content)[:90])
        elif msg.type == "ai":
            print("Finish      :", final_answer(msg.content)[:100])
    if res.get("stop_reason"):
        print("(멈춘 이유 :", res["stop_reason"], ")")

def load_hotpot(n=10):
    """HotpotQA 검증 데이터 앞 n문제 (논문과 같은 데이터셋)"""
    for ds in ["hotpotqa/hotpot_qa", "hotpot_qa"]:
        try:
            r = requests.get("https://datasets-server.huggingface.co/rows", timeout=30, params={
                "dataset": ds, "config": "distractor", "split": "validation",
                "offset": 0, "length": n}).json()
            if "rows" in r:
                return [(x["row"]["question"], x["row"]["answer"]) for x in r["rows"]]
        except Exception:
            pass
    raise RuntimeError("HotpotQA를 불러오지 못했어요 (인터넷 확인)")

############################ 7. 실험 실행 #############################
problems = load_hotpot(10)
rows, example = [], None
for i, (q, gold) in enumerate(problems, 1):
    cot_pred = run_cot(q)
    react_pred, res = run_react(q)
    row = {"q": q, "gold": gold, "cot": cot_pred, "react": react_pred,
           "cot_ok": correct(cot_pred, gold), "react_ok": correct(react_pred, gold),
           "searches": len(res["searches"]), "stop": res.get("stop_reason", "")}
    rows.append(row)
    print(f"{i:2d}. 정답 {gold[:18]:<18} | CoT {'O' if row['cot_ok'] else 'X'} | "
          f"ReAct {'O' if row['react_ok'] else 'X'} | 검색 {row['searches']}회 {row['stop']}")
    if example is None and row["react_ok"] and not row["cot_ok"]:
        example = res

n = len(rows)
cot_acc = sum(r["cot_ok"] for r in rows)
react_acc = sum(r["react_ok"] for r in rows)
avg_search = sum(r["searches"] for r in rows) / n
blocked_repeat = sum(r["stop"] == "같은 검색 반복" for r in rows)
blocked_limit = sum(r["stop"] == "검색 횟수 초과" for r in rows)

print("\n=========== 결과 ===========")
print(f"정답률   : 검색 없이(CoT) {cot_acc}/{n}  →  ReAct {react_acc}/{n}")
print(f"평균 검색 : {avg_search:.1f}회")
print(f"반복 차단 : {blocked_repeat}번 / 횟수 초과 차단 : {blocked_limit}번")

print("\n=========== CoT는 틀리고 ReAct는 맞힌 예시 ===========")
if example:
    show_trace(example)
else:
    show_trace(run_react(problems[0][0])[1])

json.dump({"n": n, "cot": cot_acc, "react": react_acc, "avg_search": avg_search,
           "blocked_repeat": blocked_repeat, "blocked_limit": blocked_limit, "rows": rows},
          open("lg12_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
