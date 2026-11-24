# CoT : 예시에 풀이 한 줄 넣으면 정답률이 오를까? (GSM8K 20문제)
# 일반 프롬프트(예시에 답만) vs CoT 프롬프트(예시에 풀이 + 답)

from langchain_openai import ChatOpenAI
import os, re, requests
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

def get_llm(max_tokens):
    return ChatOpenAI(model_name='gpt-5.6-terra', temperature=0, max_tokens=max_tokens,
                      api_key=api_key, base_url=base_url)   # lg05에서 잘 돌아간 모델

######################### 0.데이터 ###########################
rows = requests.get("https://datasets-server.huggingface.co/rows", params={
    "dataset": "openai/gsm8k", "config": "main", "split": "test",
    "offset": 0, "length": 20}).json()["rows"]
problems = [(r["row"]["question"], r["row"]["answer"].split("####")[-1].strip().replace(",", ""))
            for r in rows]

######################### 1.프롬프트 (차이는 풀이 한 줄) ###########################
EX = [("Tom has 3 boxes with 4 apples each. He eats 2 apples. How many apples are left?",
       "3 boxes x 4 apples = 12 apples. 12 - 2 = 10.", "10"),
      ("A pen costs $2 and a notebook costs $5. How much do 3 pens and 1 notebook cost?",
       "3 pens cost 3 x 2 = 6 dollars. 6 + 5 = 11.", "11")]
STANDARD = "".join(f"Q: {q}\nA: The answer is {a}.\n\n" for q, s, a in EX)
COT      = "".join(f"Q: {q}\nA: {s} The answer is {a}.\n\n" for q, s, a in EX)

def solve(prefix, q, max_tokens):
    msg = get_llm(max_tokens).invoke(prefix + f"Q: {q}\nA:")
    out = msg.content
    think = (msg.usage_metadata or {}).get("output_token_details", {}).get("reasoning", 0)
    tail = out.split("The answer is")[-1] if "The answer is" in out else out
    nums = re.findall(r"-?\d+\.?\d*", tail.replace(",", ""))
    pred = (nums[0] if "The answer is" in out else nums[-1]).rstrip(".") if nums else ""
    return out, pred, think

######################### 2.실행 ###########################
std = cot = 0
std_think = cot_think = 0                     # 모델이 '속으로' 생각한 토큰 수
example = None
for i, (q, gold) in enumerate(problems, 1):
    s_out, s_pred, s_t = solve(STANDARD, q, 4000)   # 일반 : 예시에 답만
    c_out, c_pred, c_t = solve(COT, q, 4000)        # CoT : 예시에 풀이 + 답
    std_think += s_t; cot_think += c_t
    std += s_pred == gold
    cot += c_pred == gold
    print(f"{i:2d}. 정답 {gold:>6} | 일반 {s_pred or '(빈칸)':>6} | CoT {c_pred or '(빈칸)':>6}")
    if s_pred != gold and c_pred == gold and not example:
        example = (q, gold, s_out, c_out)       # 슬라이드에 넣을 예시 하나

print(f"\n일반 {std}/{len(problems)}  →  CoT {cot}/{len(problems)}")
print(f"속으로 생각한 토큰 (문제당 평균) : 일반 {std_think/len(problems):.0f}개 / CoT {cot_think/len(problems):.0f}개")
if example:
    q, gold, s_out, c_out = example
    print("\n[일반은 틀리고 CoT는 맞힌 문제]\n문제 :", q, "\n정답 :", gold)
    print("\n[일반 답]", s_out, "\n\n[CoT 답]", c_out)
