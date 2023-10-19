from typing import Annotated
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from reference.wpc import wpc
app = FastAPI()

#CORS(https://www.jasonchoi.dev/posts/fastapi/cors-allow-setting)
origins = [
    "http://localhost"
]

# 미들웨어 추가 -> CORS 해결위해 필요(https://ghost4551.tistory.com/46)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/process')
def process(p_id:str,code:Annotated[str, Body(embed=True)]):
    try:
        fixed_code =wpc.process(code,p_id)
    except Exception as e:
        raise HTTPException(500,e)
    return {'fixed_code':fixed_code}

@app.get('/hello')
def process():
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
    except:
        return False
    return True