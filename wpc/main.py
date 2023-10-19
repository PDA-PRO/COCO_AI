import json
from typing import Annotated
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from reference.wpc import wpc
app = FastAPI()

origins = [
    "http://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Wpc_response(BaseModel):
    fixed_code:str
    bug_code:str

@app.post('/process',response_model=Wpc_response)
def process(p_id:str,code:Annotated[str, Body(embed=True)]):
    """
    모델 추론
    p_id를 풀이하는 raw code중 틀린 곳을 고친 코드를 리턴

    사용가능한 문제 리스트는 

    here: https://github.com/PDA-PRO/COCO_AI/tree/main/wpc 의 p_list.txt를 참고하시길 바랍니다.

    params
    - p_id : 문제 번호 ex) "p00001"
    - code : raw code
    -------------------------------------------
    return
    - fixed_code : 틀린 곳을 고친 코드
    - bug_code : 원래의 버그가 있는 코드
    """
    try:
        fixed_code,generalized_code =wpc.process(code,p_id)
    except Exception as e:
        raise HTTPException(500,e)
    return {'fixed_code':fixed_code,'bug_code':generalized_code}

@app.get('/hello')
def test():
    """
    모델일 정상적으로 동작하는지 테스트

    returns
    - 정상동작시 True, 비정상동작시 False
    """

    try:
        wpc.process('''
a,b,c=map(int,input().split())
sound=0
for i in range(c):
    b-=a
if a<0:
    break
else:
    sound+=1
print(sound)''',"p03105")
    except Exception as e:
        print(str(e))
        return False
    return True

@app.get('/problem')
def p_detail(p_id:str):
    """
    알고리즘 문제 상세 정보 조회, 
    사용가능한 문제 리스트는 

    here: https://github.com/PDA-PRO/COCO_AI/tree/main/wpc 의 p_list.txt를 참고하시길 바랍니다.

    params
    - p_id : 문제 번호 ex) "p00001"
    -------------------------------------------
    returns
    - 정상동작시 True, 비정상동작시 False
    """
    with open("/".join(__file__.split('/')[:-1])+'/task_detail.json',"r",encoding="utf8") as file:
        detail:dict=json.load(file)
        return detail.get(p_id)