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

######################## 1. State 정의 ########################
class State(TypedDict):
    messages : Annotated[list, add_messages]

######################## 2. Node 정의 ########################
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
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# ############################ 6. 그래프실행 #############################

# 두 질문에서 같은 thread_id를 사용해 대화 기록을 이어갑니다.
config = {"configurable": {"thread_id": "memory-study-session"}}

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