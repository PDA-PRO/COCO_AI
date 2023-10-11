# COCO_AI
충북대학교 소프트웨어학과 졸업작품 COCO : Coding Coach - 조민수, 서종원, 김광모

## **1\. 소개**

### 1) 소개

[COCO : COdingCOach | 코딩 초보자들을 위한 온라인 저지 | ~ing](https://sjw-record.tistory.com/16)

졸업작품에서 알고리즘 문제의 오답 코드에서 틀린 부분을 찾아주는 기능을 구현하기 위해 AI를 활용했습니다.

졸업작품 내에서 이 기능을 WPC(Wrong Part of code)라고 명명합니다.

### 2) AI의 중요성과 활용 가능성 간단히 언급

알고리즘 문제 풀이 중 자신이 작성한 코드 내에서 논리적으로 잘못된 곳으로 인해 오답이 되었을 때 스스로 제출 코드 내에서 어떤 부분이 잘못된 것인지 찾고 고치지 못하는 것을 방지하기 위해 이 모델을 개발하게 되었습니다.

졸업작품에서는 쉬운 수준의 알고리즘 문제에 대한 제출 코드의 논리적 오류를 찾는 것으로 범위를 제한했지만 데이터셋이 더 커진다면 더 다양한 알고리즘 문제에 대해 활용할 수 있을 것이라고 생각됩니다.

---

## **2\. 문제 정의와 목표 설정**

### 1) 문제 정의

알고리즘 문제에 대한 오답 코드에서 문법오류, 런타임 오류에 비해 논리적 오류는 알고리즘 문제별로 발생하는 이유가 다르기 때문에 사람이 직접 코드를 디버깅하여 오류 위치를 찾아 고쳐야합니다. 하지만 코딩을 자주 접하지 못한 초보자들에게는 자신의 코드를 디버깅하고 논리적 오류 위치를 찾는 것은 상당히 어려운 일입니다.

자신의 코드의 오류 위치를 찾지 못하는 일이 반복된다면 코딩에 흥미를 잃게 될 수도 있습니다.

### 2) 목표 설정

따라서, 위 문제를 해결하기 위해 알고리즘 문제와 오답코드를 입력값으로 받아 오답 코드의 논리적 오류를 고친 코드를 출력값으로 하는 AI 모델을 만들기로 했습니다.우선 모델 입력값의 범위를 초보자가 풀만한 쉬운 수준의 문제로만 한정하였습니다.

---

## **3\. AI 기술 선택 및 활용 방법**

### 1) 모델 선택

CodeBERT - 6개 프로그래밍 언어(Python, Java, JavaScript, PHP, Ruby, Go)의 NL-PL 쌍에 대해 사전 훈련된 다중 프로그래밍 언어 모델

GraphCodeBERT - CodeBERT가 소스 코드의 semantic-level structure에 대한 고려 없이 소스 코드를 token들의 시퀸스로만 표현하는 것을 개선하고자 Data Flow를 이용해 코드의 고유 구조(inherent structure)를 고려한 모델

[https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/refinement](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/refinement)

의 Seq2Seq 모델을 약간 수정하여 미세조정

### 2) 모델 구조

![wpc 모델 구조 drawio](https://github.com/PDA-PRO/COCO_AI/assets/80380576/cf923eac-dcf8-4a49-b3e1-9052bb4a3bbb)


왼쪽의 기존 구조에서 알고리즘 문제의 설명문을 입력값으로 추가하여 각 알고리즘 문제에 대한 논리적 오류들을 학습하도록 하였다.

---

## **4\. 데이터 수집과 전처리**

### 1) 데이터 수집

\- algorithm problem description

[https://developer.ibm.com/exchanges/data/all/project-codenet/](https://developer.ibm.com/exchanges/data/all/project-codenet/) 

IBM Research에서 만든 프로그래밍과 소프트웨어 개발에 관련된 대규모 데이터셋, 다양한 알고리즘 문제와 각 문제에 제출된 다양한 언어의 코드로 이루어져있다.

\- bug code - fixed code 쌍 데이터

  
FixEval: Execution-based Evaluation of Program Fixes for Competitive Programming Problems 의 데이터 셋을 활용  
codenet의 데이터셋중에서 파이썬으로 작성한 코드만 뽑아내고 한 유저가 한가지 문제를 풀어내기까지의 코드 변화내역을 1:m으로 엮은 데이터셋   

### 2) 데이터 전처리

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM,RobertaTokenizer
import json
import os
from bs4 import BeautifulSoup as bs
import ast
import re
import argparse
from tree_sitter import Language, Parser
import random

JSON_LEN=116
EXTRACT_DESC_PATH="tmp_p_desc/"
method=0
var=0

#토큰화에 필요한 tree_sitter 준비
Language.build_library(
# Store the library in the `build` directory
'build/my-languages.so',
# Include one or more languages
[
    'tree-sitter-python'
]
)
PY_LANGUAGE = Language('build/my-languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

def remove_multi_line_comment(old_code):
    p = re.compile('(\"\"\")')
    swt=True
    word_range=[]
    for i in p.finditer(old_code):
        if swt:
            word_range.append(i.span()[0])
            swt=False
        else:
            word_range.append(i.span()[1])
            swt=True
    if len(word_range):
        new_code=""
        new_code+=old_code[:word_range[0]]
        for i in range(1,len(word_range)-1,2):
            new_code+=old_code[word_range[i]:word_range[i+1]]
        new_code+=old_code[word_range[-1]:]
        return new_code
    else:
        return old_code

def abstract_pl(code,ident):
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        def dfs(node):
            global method,var
            if node.child_count==0:
                if node.type=='identifier':
                    if node.parent.type not in ["for_in_clause","call","attribute","dotted_name","except_clause"]:
                        if (not ident.get(node.text)) and (node.text not in [b'i',b'j',b'k',b'__name__',b'int',b'str',b'input',b'float']):
                            if node.parent.type=="function_definition":
                                ident[(node.text)]="method"+str(method)
                                method+=1
                            else:
                                ident[(node.text)]="var"+str(var)
                                var+=1
                return
            else:
                for i in range(node.child_count):
                    dfs(node.children[i])
        dfs(root_node)
        for i in ident.items():
            code=re.sub(r'\b'+i[0].decode()+r'\b',i[1] , code)
        
        return code,ident

def main():
    global method,var
    parser = argparse.ArgumentParser()

    parser.add_argument("--code_pair_path", default=None, type=str, required=True,
                        help="code pair dataset path")
    parser.add_argument("--p_desc_path", default=None, type=str, required=True,
                        help="problem descriptions path")
    parser.add_argument("--max_token_len", default=256, type=int, required=False,
                        help="max token length")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    args = parser.parse_args()
    CODE_PAIR_PATH=args.code_pair_path
    DESC_PATH=args.p_desc_path
    no_desc_id=["p02479","p02480","p02481","p02482","p02483","p02484","p02485","p02486","p02487","p02488","p02489","p02490","p02491","p02492","p02493","p02494","p02495","p02496","p02497","p02498","p02499","p02506","p02510","p02523","p02524","p02525","p02526","p02527","p02528","p02529","p02530","p02531","p02532"]
    p_num=0
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

    #WA 코드쌍 추출 - 논리적 오류가 존재하여 TC를 통과하지 못한 코드
    print("extract WA code pair")
    for i in range(JSON_LEN):
        new_json=[]
        if i%10==0:
            print(i)
        with open(CODE_PAIR_PATH+str(i)+'.json', 'r') as f:
            json_data = json.load(f)
        for j in range(len(json_data)):
            if json_data[j]:
                if json_data[j][0]["verdict"]=="Wrong Answer" and (json_data[j][0]["problem_id"] not in no_desc_id):
                    temp_code_pair=[]
                    try:
                        for k in range(2):
                            node=ast.parse(json_data[j][k]["code_tokens"])
                            new_code=ast.unparse(node).replace("    ","\t")
                            new_code=remove_multi_line_comment(new_code)
                            temp_code_pair.append(
                                {
                                    "problem_id": json_data[j][k]["problem_id"],
                                    "code_tokens": new_code,
                                    "submission_id": json_data[j][k]["submission_id"],
                                }
                            )
                    except:
                            continue
                    new_json.append(temp_code_pair)
                    p_num+=1
        with open(CODE_PAIR_PATH+str(i)+'.json', 'w',encoding='utf-8') as mkf:
            json.dump(new_json, mkf, indent="\t")
    print("total WA code pair num : ",p_num)

    
    #문제중 score가 400이상인 문제 삭제 및 설명부분 추출 - 초보자에게 쉬운 문제로 범위 제한
    print("prepare problem_descriptions")
    p_list=os.listdir(DESC_PATH)
    if not os.path.exists(EXTRACT_DESC_PATH):
        os.mkdir(EXTRACT_DESC_PATH)
    for i in p_list:
        if int(i.split(".")[0][1:])>=2479:
            with open(DESC_PATH+i) as html:
                soup = bs(html, "html.parser")
                try:
                    elements = soup.var.text
                    if int(elements)<=300:
                        text=""
                        for j in soup.find_all("section")[0].contents:
                            if j.text.lower()=="input":
                                break
                            text+=j.text
                        text=text.replace("Problem Statement","").strip()
                        with open(EXTRACT_DESC_PATH+i.split(".")[0]+".txt","w") as desc:
                            desc.write(text)
                    else:
                        os.remove(DESC_PATH+i)
                except:
                    text=""
                    if soup.find_all("section"):
                        for j in soup.find_all("section")[0].contents:
                            if j.text.lower()=="input":
                                break
                            text+=j.text
                        text=text.replace("Problem Statement","").strip()
                        with open(EXTRACT_DESC_PATH+i.split(".")[0]+".txt","w") as desc:
                            desc.write(text)
        else:
            with open(DESC_PATH+i) as html:
                soup = bs(html, "html.parser")
                text=""
                for j in soup.contents:
                    if j.text.lower()=="input":
                        break
                    if j.name!="script":
                        text+=j.text
                text=text.split("Input")[0].replace("$","").strip()
                with open(EXTRACT_DESC_PATH+i.split(".")[0]+".txt","w") as desc:
                    desc.write(text)

    
    #특수기호 및 공백제거 - 문제별로 같은 의미지만 다른 유니코드의 기호들을 변환
    p_list=os.listdir(EXTRACT_DESC_PATH)
    s_symbol={"≠":"!=","\dots":"...", "\cdot":"...", "\ldots":"...", "…":"...","\leq":"<=", "\le":"<=", "≤":"<=", "≦":"<=", "\times": "*","\ge":">=", "\geq":">=","\{": "{","\}": "{"  }
    for i in p_list:
        p_text=""
        with open(EXTRACT_DESC_PATH+i,"r") as txt:
            for j in txt.readlines():
                if j=="\n":
                    continue
                p_text+=j.rstrip()+"\n"
            for j in s_symbol.items():
                p_text=p_text.replace(j[0],j[1])
        with open(EXTRACT_DESC_PATH+i,"w") as txt:
            txt.write(p_text)

    

    #코드쌍 추상화 - 변수와 함수의 이름을 var1 ... varX, method1 ... methodX 로 추상화
    print("abstracte code pair")
    for i in range(JSON_LEN):
        if i%10==0:
            print(i)
        new_json=[]
        with open(CODE_PAIR_PATH+str(i)+'.json', 'r') as f:
            json_data = json.load(f)
            for j in range(len(json_data)):
                code_pair=json_data[j]
                temp_ident={}
                method=0
                var=0
                for k in range(2):
                    code=code_pair[k]["code_tokens"]
                    new_code,temp_ident=abstract_pl(code,temp_ident)
                    code_pair[k]["code_tokens"]=new_code
                new_json.append(code_pair)
            with open(CODE_PAIR_PATH+str(i)+'.json', 'w',encoding='utf-8') as mkf:
                json.dump(new_json, mkf, indent="\t")

    #nl과 pl 토큰 길이 추출
    print("extract nl-pl token length : max_token_len ",args.max_token_len)
    removed_num=0
    code_num=0
    desc_list=os.listdir(EXTRACT_DESC_PATH)
    for i in range(len(desc_list)):
        desc_list[i]=desc_list[i].split(".")[0]
    if not os.path.exists(CODE_PAIR_PATH):
        os.mkdir(CODE_PAIR_PATH)
    for i in range(JSON_LEN):
        new_json=[]
        if i%10==0:
            print(i)
        with open(CODE_PAIR_PATH+str(i)+'.json', 'r') as f:
            json_data = json.load(f)
            for j in range(len(json_data)):
                code_pair=json_data[j]
                not_compile=False
                for k in range(2):
                    code=code_pair[k]["code_tokens"]
                    code_token=tokenizer.tokenize(code)
                    nl_token=[]
                    if code_pair[k]["problem_id"] in desc_list:
                        with open(EXTRACT_DESC_PATH+code_pair[k]["problem_id"]+".txt",'r') as desc:
                            nl_token=tokenizer.tokenize(desc.read())
                            code_pair[k]["nl_pl_token"]=len(code_token)+len(nl_token)
                            if len(code_token)+len(nl_token)>args.max_token_len-3:
                                not_compile=True
                    else:
                        not_compile=True
                if not_compile:
                    removed_num+=1
                    continue
                else:
                    new_json.append(code_pair)
                    code_num+=1
        with open(CODE_PAIR_PATH+str(i)+'.json', 'w',encoding='utf-8') as mkf:
            json.dump(new_json, mkf, indent="\t")
    print("usable code pair num : ",code_num,"removed num : ",removed_num)  

    #문제별 코드 수 
    print("clac code pair count per problem")
    temp={}
    for i in range(JSON_LEN):
        with open(CODE_PAIR_PATH+str(i)+'.json', 'r') as f:
            json_data = json.load(f)
            for j in range(len(json_data)):
                p_id=json_data[j][0]["problem_id"]
                try:
                    temp[p_id]+=1
                except:
                    temp[p_id]=1
    with open('nl_pl_p_num.json', 'w',encoding='utf-8') as mkf:
        json.dump(temp, mkf, indent="\t")

    #문제별 코드 수가 100 미만인 코드쌍 및 문제 설명 삭제 - 학습 코드의 양이 부족한 문제 제거
    print("remove code pair and problem description less than count 100")
    removed_num=0
    code_num=0
    with open('nl_pl_p_num.json', 'r') as fff:
        p_num = json.load(fff)
        temp={}
        for i in range(JSON_LEN):
            new_json=[]
            print(i)
            with open(CODE_PAIR_PATH+str(i)+'.json', 'r') as f:
                json_data = json.load(f)
                for j in range(len(json_data)):
                    code_pair=json_data[j]
                    p_id=code_pair[0]["problem_id"]
                    if p_num[p_id]>=100:
                        new_json.append(code_pair)
                        code_num+=1
                    else:
                        if os.path.exists(EXTRACT_DESC_PATH+p_id+".txt"):
                            os.remove(EXTRACT_DESC_PATH+p_id+".txt")
                        removed_num+=1
            with open(CODE_PAIR_PATH+str(i)+'.json', 'w',encoding='utf-8') as mkf:
                json.dump(new_json, mkf, indent="\t")
        print("remain count : ",code_num,"remove count : ",removed_num)

    #nl + pl 데이터셋 생성
    print("generate nl_pl pair dataset")
    desc_list=os.listdir(EXTRACT_DESC_PATH)
    data={}
    for i in range(len(desc_list)):
        data[desc_list[i].split(".")[0]]=[]

    for i in range(JSON_LEN):
        with open(CODE_PAIR_PATH+str(i)+'.json', 'r') as f:
            f.read
            json_data = json.load(f)
            for j in range(len(json_data)):
                code_pair=json_data[j]
                buddy_code=code_pair[0]["code_tokens"]
                fixed_code=code_pair[1]["code_tokens"]
                data[code_pair[0]["problem_id"]].append({
                    "buggy_code":remove_multi_line_comment(buddy_code),
                    "fixed_code":remove_multi_line_comment(fixed_code)
                })
    train=[]
    test=[]
    val=[]
    for p_name,code in data.items():
        random.shuffle(code)
        for i in range(len(code)):
            if i<len(code)*0.7:
                train.append({"p_name":p_name,"code":code[i]})
            elif i<len(code)*0.9:
                test.append({"p_name":p_name,"code":code[i]})
            else:
                val.append({"p_name":p_name,"code":code[i]})
    
    if not os.path.exists("data"):
        os.mkdir("data")

    with open("data/train.json","w") as data_set:
        json.dump(train,data_set,indent="\t")
    with open("data/test.json","w") as data_set:
        json.dump(test,data_set,indent="\t")
    with open("data/val.json","w") as data_set:
        json.dump(val,data_set,indent="\t")
        
if __name__ == "__main__":
    main()
```

---

## **5\. 모델 훈련**

### 1) 모델 훈련

```bash
python ./COCO_AI/graphcodebert_0718/run.py \
--do_train \
--do_eval \
--model_type roberta \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--train_filename /content/drive/MyDrive/coco_ai/train.json \
--dev_filename /content/drive/MyDrive/coco_ai/val.json \
--p_desc_path /content/drive/MyDrive/coco_ai/tmp_p_desc \
--output_dir /content/drive/MyDrive/coco_ai/graphcodebert_256_0724/output \
--load_model_path /content/drive/MyDrive/coco_ai/graphcodebert_256_0724/output/checkpoint-last/pytorch_model.bin \
--max_source_length 512 \
--max_target_length 256 \
--beam_size 10 \
--train_batch_size 32 \
--eval_batch_size 32 \
--learning_rate 5e-5 \
--num_train_epochs 20 2>&1| tee /content/drive/MyDrive/coco_ai/graphcodebert_256_0724/output/train.log
```

### 2) 모델 훈련 과정

Google Colab A100 고용량 RAM 을 사용하여 10epoch 씩 총 20epoch 학습 수행

더보기

.......

\*\*\*\*\* Running evaluation \*\*\*\*\*  
09/10/2023 07:43:18 - INFO - \_\_main\_\_ -     Num examples = 25381  
09/10/2023 07:43:18 - INFO - \_\_main\_\_ -     Batch size = 32  
09/10/2023 07:47:45 - INFO - \_\_main\_\_ -     eval\_ppl = 1.1595  
09/10/2023 07:47:45 - INFO - \_\_main\_\_ -     global\_step = 55981  
09/10/2023 07:47:45 - INFO - \_\_main\_\_ -     train\_loss = 0.029  
09/10/2023 07:47:45 - INFO - \_\_main\_\_ -     \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
09/10/2023 08:08:31 - INFO - \_\_main\_\_ -     bleu-4 = 73.64   
09/10/2023 08:08:31 - INFO - \_\_main\_\_ -     xMatch = 37.0   
09/10/2023 08:08:31 - INFO - \_\_main\_\_ -     \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
09/10/2023 08:08:31 - INFO - \_\_main\_\_ -     Best BLEU+xMatch:110.64  
09/10/2023 08:08:31 - INFO - \_\_main\_\_ -     \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
epoch 10 loss 0.0338: 100%|██████████| 5598/5598 \[1:38:14<00:00,  1.05s/it\]

\*\*\*\*\* Running evaluation \*\*\*\*\*

09/10/2023 09:46:47 - INFO - \_\_main\_\_ - Num examples = 25381

09/10/2023 09:46:47 - INFO - \_\_main\_\_ - Batch size = 32

09/10/2023 09:51:14 - INFO - \_\_main\_\_ - eval\_ppl = 1.15586

09/10/2023 09:51:14 - INFO - \_\_main\_\_ - global\_step = 61579

09/10/2023 09:51:14 - INFO - \_\_main\_\_ - train\_loss = 0.0338

09/10/2023 09:51:14 - INFO - \_\_main\_\_ - \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

09/10/2023 10:11:48 - INFO - \_\_main\_\_ - bleu-4 = 73.29

09/10/2023 10:11:48 - INFO - \_\_main\_\_ - xMatch = 36.8

09/10/2023 10:11:48 - INFO - \_\_main\_\_ - \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

epoch 11 loss 0.0305: 80%|███████▉ | 4470/5598 \[1:18:27<19:47, 1.05s/it\]

---

## **6. 결과 분석 및 평가**

### 1) 평가 지표

code refinement영역에서 평가지표는 BLEU, CodeBLEU, Acc, TestCase-base평가 등 여러가지가 존재하지만 정확한 평가지표가 없기 때문에 가장 많은 평가지표로 활용된 BLEU, CodeBLEU, Acc를 활용

BLEU-4 : 73.64

xMatch : 37.0

### 2) 테스트 데이터 모델 성능 평가

문제 설명과 오답코드를 입력값으로 넣고 나온 출력코드를 해당 알고리즘 문제의 TC로 채점하여 맞은 비율을 측정

---

## **8\. 참고문헌**

[https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/refinement](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/refinement)

[https://github.com/IBM/Project\_CodeNet](https://github.com/IBM/Project_CodeNet)

[https://arxiv.org/abs/1812.08693](https://arxiv.org/abs/1812.08693)

[https://github.com/mahimanzum/FixEval](https://github.com/mahimanzum/FixEval)
