import faiss
import numpy
import pymysql
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import (RobertaTokenizer)

tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

max_token_len=512
clustering_k=10

def clustering(task_id:int,sub_id:int):
    con = pymysql.connect(host=os.getenv('DATABASE_HOST'), user=os.getenv('DATABASE_USERNAME'), password=os.getenv('DATABASE_PASSWORD')\
                          ,port=int(os.getenv('DATABASE_HOST')),db=os.getenv('DATABASE'), charset='utf8')  # 한글처리 (charset = 'utf8')
    cur = con.cursor(pymysql.cursors.DictCursor)
    sql='SELECT sub.code FROM coco.sub_ids as ids, coco.submissions as sub where ids.task_id=%s and sub.id=ids.sub_id and sub.status=3;'
    cur.execute(sql,[task_id])
    result=cur.fetchall()
    my_code_sql=sql='SELECT sub.code FROM coco.sub_ids as ids, coco.submissions as sub where ids.task_id=%s and sub.id=%s'
    cur.execute(my_code_sql,[task_id,sub_id])
    my_code_result=cur.fetchall()[0]
    code_list=[my_code_result['code']]
    for i in result:
        code_list.append(i['code'])

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
        return_list.append({'code':code_list[i[0][0]],'distance':float(i[1][0])})
            
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
def process(task_id:int,sub_id:int):
    '''
    해당 문제의 다른 로직의 코드 조회

    params
    - task_id
    - sub_id
    -------------------------
    returns
    - 코드 리스트
        - code : 코드
        - distance : 요청한 코드와 응답 코드의 차이
    '''
    try:
        return_list=clustering(task_id,sub_id)
        return return_list
    except Exception as e:
        raise HTTPException(500,str(e))

@app.get('/hello')
def ready():
    return True

@app.put('/config')
def update_config(max_token:int,k:int):
    '''
    faiss 설정 변경

    params
    - max_token : 코드의 최대 길이
    - k : k mean 클러스터링의 파라미터 
    '''
    global max_token_len,clustering_k
    max_token_len=max_token
    clustering_k=k
    return 1

@app.get('/config')
def read_config():
    '''
    faiss 설정 조회

    returns
    - max_token_len : 코드의 최대 길이
    - clustering_k : k mean 클러스터링의 파라미터 
    '''
    global max_token_len,clustering_k
    return {'max_token_len':max_token_len,'clustering_k':clustering_k}