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

    #WA 코드쌍 추출
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

    
    #문제중 score가 400이상인 문제 삭제 및 설명부분 추출
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

    
    #특수기호 및 공백제거
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

    

    #코드쌍 추상화
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

    #문제별 코드 수가 100 미만인 코드쌍 및 문제 설명 삭제
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