#Tool calling : 외부기능을 가져와서 사용하기!! 
#https://app.tavily.com/home

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"
llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra', 
    # model_name = 'gpt-5-nano', 
    temperature= 0,
    max_tokens = 1000,
    api_key= api_key,
    base_url= base_url,
)

######################### 0.Tool 정의 ###########################
from langchain_tavily import TavilySearch

#웹 검색 도구 생성 
web_search = TavilySearch(
    max_results=3,
)

# 테스트용 검색 (확인 끝났으니 주석 처리)
# result = web_search.invoke({
#     "query" : "삼성전자의 오늘주가 알려줘 "
# })
# print(result)
# print(result['results'][-1]['content'])

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
#메모리 추가 첫번쨰 
from langgraph.checkpoint.memory import InMemorySaver

######################## 1. State 정의 ########################
class State(TypedDict):
    messages: Annotated[list, add_messages]
    search_result: str
    analysis_result: str
    final_result: str

######################## 2. Node 정의 ########################
# 검색 에이전트 
def search_agent(state: State):
    print("검색 에이전트 실행")
    question = state["messages"][-1].content

    result = web_search.invoke({
        "query": question
    })
    return {"search_result": str(result)}

def analysis_agent(state:State):
    print("분석 에이전트 실행")
    question = state["messages"][-1].content

    response= llm.invoke(
        f"다음 질문을 기존 지식을 바탕으로 분석해줘.\n{question}"
    )
    return {"analysis_result": response.content}
def merge_agent(state:State):
    result = (
        state["search_result"] + "\n\n" +
        "=====================================" + "\n" +
        state["analysis_result"]
    )
    return {"final_result": result}


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

#################### 3. Graph 구성, Node 추가 ####################
# 그래프 생성
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

# tool_node = ToolNode(tools)
# graph_builder.add_node("tools", tool_node)
graph_builder.add_node("tools", ToolNode(tools))

####################4. 엣지추가 #############################
graph_builder.add_edge(START, "chatbot")

# graph_builder.add_edge("chatbot","tools")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition #tool_calls 이 있나?
    #tool_calls 가 있으면 'tools'반환
    #tool_calls 가 없으면 END 반환
    #tools_condition END포함 됫지롱
)

# 검색 결과를 챗봇에게 다시 넘겨서 답변을 쓰게 함 (END → chatbot)
graph_builder.add_edge("tools", "chatbot")

############################## 5.컴파일 #################################

#메모리 추가 두번째
memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)

# ############################ 6. 그래프실행 #############################

#메모리 추가 세번째
# 두 질문에서 같은 thread_id를 사용해 대화 기록을 이어갑니다.
config = {"configurable": {"thread_id": "user01"}}

# 첫번째 질문
response = graph.invoke(
    {"messages": [("user", "내 이름은 기요미야.")]},
    config=config,
)

print(response["messages"][-1].content)

print("=================== 두번째 질문 ==========================")
# 두번째 질문
response = graph.invoke(
    {"messages": [("user", "내 이름이 뭐라고 했지?")]},
    config=config,
)
print(response["messages"][-1].content)

#5. 그래프 구성 
builder = StateGraph(State)

builder.add_node("search_agent", search_agent)
builder.add_node("analysis_agent", analysis_agent)
builder.add_node("merge_agent", merge_agent)

####################### 병렬 실행 #############################
builder.add_edge(START, "search_agent")
builder.add_edge(START, "analysis_agent")

########################### 변환 #########################
builder.add_edge(["search_agent", "analysis_agent"], "merge_agent")

builder.add_edge("merge_agent", END)

#컴파일 
graph = builder.compile()

################### 그래프 실행###########################
query = " 엔비디아의 미래 전망에 대해 알려줘 "

response= graph.invoke(
    {"messages":[("user",query)]}
)

print("Assistant : ", response["messages"][-1].content)






#실행 
response = graph.invoke({
    "messages": [("user", "엔비디아의 미래 전망은?")]
})

print(response["final_result"])

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()
img = Image.open(BytesIO(png_data))

plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()


