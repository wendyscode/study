# 실무에서는 다음과 같은 기능을 여러 함수에 공통으로 적용할 때 유용하다.
# 1. 함수 실행시간 측정
# 2. 로그 기록
# 3. 사용자 권한 검사
# 4. 예외 처리
# 5. 함수 호출 횟수 기록


import time

def timer(func):
    def wrapper():
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        print("걸린시간: ", round(end-start, 6), '초')
    return wrapper

@timer
def hello():
    print("안녕하세요")
    time.sleep(1)

hello()

@timer
def study():
    print("NP공부중이닷")
    time.sleep(1.5)

study()