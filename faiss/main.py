import faiss
import json
import numpy
import pymysql
from typing import Annotated
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import (RobertaTokenizer)


tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

max_token_len=512
clustering_k=10

def clustering(task_id:int,code_in:str):
    con = pymysql.connect(host='localhost', user='root', password='',port=3307,db='coco', charset='utf8')  # 한글처리 (charset = 'utf8')
    cur = con.cursor(pymysql.cursors.DictCursor)
    sql='SELECT sub.code FROM coco.sub_ids as ids, coco.submissions as sub where ids.task_id=%s and sub.id=ids.sub_id and sub.status=3;'
    cur.execute(sql,[task_id])
    result=cur.fetchall()
    code_list=[code_in]
    for i in result:
        code_list.append(i['code'])

    with open('./faiss/ttt.json',"r",encoding='utf8') as ss:
        temp=json.load(ss)
        code_list.extend(temp)
    #0 패딩
    code_ids_list=[]
    for i in code_list:
        tokenized_code=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(i))
        if len(tokenized_code)>=max_token_len:
            continue
        else:
            tokenized_code=tokenized_code+(max_token_len-len(tokenized_code))*[0]
            code_ids_list.append(tokenized_code)

    #numpy로 변환 및 0-1정규화
    code_ids_list=numpy.asarray(code_ids_list, dtype='float32')
    min_v=numpy.min(code_ids_list)
    max_v=numpy.max(code_ids_list)
    for i in range(len(code_ids_list)):
        for j in range(max_token_len):
            code_ids_list[i][j]=(code_ids_list[i][j]-min_v)/(max_v-min_v)
    #인덱스 생성
    kmeans = faiss.Kmeans(max_token_len,clustering_k)
    kmeans.train(code_ids_list)

    index = faiss.IndexFlatL2(max_token_len)
    index.add(code_ids_list)
    D, I = index.search(kmeans.centroids, 1) 
    return_list=[]
    for i in zip(I,D):
        return_list.append({'code':code_list[i[0][0]],'distance':i[1][0]})
            
    return return_list

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
def process(task_id:int,code:Annotated[str, Body(embed=True)]):
    try:
        return_list=clustering(task_id,code)
        return return_list
    except Exception as e:
        raise HTTPException(500,str(e))

@app.get('/hello')
def ready():
    return True