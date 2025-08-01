# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import ast
import re
import tree_sitter_python as tspython
import os
import torch
import random
import logging
import numpy as np
from io import open
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from .parser.DFG import DFG_python
from .parser.utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token)
from .config import AiConfig
from .model import Seq2Seq

class WPC():
    def __init__(self):
        #로거 설정
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
        self.logger = logging.getLogger(__name__)

        #폴더경로 추출
        self.folder_path='/'.join(__file__.split("/")[:-2])

        #data flow graph 생성에 필요한 tree_parser 설정
        PY_LANGUAGE = Language(tspython.language())
        self.tree_parser = Parser(PY_LANGUAGE)

        #코드 추상화에 필요한 변수
        self.method=0
        self.var=0
        
        # Setup CUDA, GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        # Set seed
        random.seed(42)
        os.environ['PYHTONHASHSEED'] = str(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

        #베이스 모델및 설정, 토크나이저 설정
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
        config = config_class.from_pretrained("microsoft/graphcodebert-base")
        self.tokenizer = tokenizer_class.from_pretrained("microsoft/graphcodebert-base")

        #budild model
        encoder = model_class.from_pretrained("microsoft/graphcodebert-base",config=config)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                    beam_size=AiConfig.beam_size,max_length=AiConfig.max_target_length,
                    sos_id=self.tokenizer.cls_token_id,eos_id=self.tokenizer.sep_token_id)

        #모델 불러오기
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.folder_path+"/pytorch_model.bin"),strict=False)
        else:
            self.model.load_state_dict(torch.load(self.folder_path+"/pytorch_model.bin",map_location='cpu'),strict=False)

        self.model.to(self.device)
        if self.n_gpu > 1:
            # multi-gpu training
            self.model = torch.nn.DataParallel(self.model)

    def extract_dataflow(self,code):
        """
        remove comments, tokenize code and extract dataflow
        """
        #remove comments
        try:
            code=remove_comments_and_docstrings(code)
        except:
            pass
        #obtain dataflow
        try:
            tree = self.tree_parser.parse(bytes(code,'utf8'))
            root_node = tree.root_node
            tokens_index=tree_to_token_index(root_node)
            code=code.split('\n')
            code_tokens=[index_to_code_token(x,code) for x in tokens_index]
            index_to_code={}
            for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
                index_to_code[index]=(idx,code)
            try:
                DFG,_=DFG_python(root_node,index_to_code,{})
            except:
                DFG=[]
            DFG=sorted(DFG,key=lambda x:x[1])
            indexs=set()
            for d in DFG:
                if len(d[-1])!=0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_DFG=[]
            for d in DFG:
                if d[1] in indexs:
                    new_DFG.append(d)
            dfg=new_DFG
        except:
            dfg=[]
        return code_tokens,dfg
    
    class Example(object):
        """A single training/test example."""
        def __init__(self,
                    source,
                    target,
                    p_id
                    ):
            self.source = source
            self.target = target
            self.p_id=p_id


    class InputFeatures(object):
        """A single training/test features for a example."""
        def __init__(self,
                    example_id,
                    source_ids,
                    position_idx,
                    dfg_to_code,
                    dfg_to_dfg,
                    target_ids,
                    source_mask,
                    target_mask,

        ):
            self.example_id = example_id
            self.source_ids = source_ids
            self.position_idx = position_idx
            self.dfg_to_code = dfg_to_code
            self.dfg_to_dfg = dfg_to_dfg
            self.target_ids = target_ids
            self.source_mask = source_mask
            self.target_mask = target_mask

    ## 2023.7 모델 입력 임베딩으로 문제 설명 추가
    def convert_examples_to_features(self,examples, tokenizer):
        """
        모델에 입력값으로 쓸 수 있도록 임베딩
        """
        total_source_len=0
        features = []
        for example_index, example in enumerate(tqdm(examples,total=len(examples))):
            ##extract data flow
            code_tokens,dfg=self.extract_dataflow(example.source)
            code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
            ori2cur_pos={}
            ori2cur_pos[-1]=(0,0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))
            code_tokens=[y for x in code_tokens for y in x]
            total_source_len+=len(code_tokens)

            #problem description
            p_desc_tokens=[]
            with open(self.folder_path+"/desc/"+example.p_id+".txt","r") as p_desc_file:
                p_desc_tokens=tokenizer.tokenize(p_desc_file.read())
            p_desc_length=len(p_desc_tokens)+1
            total_source_len+=p_desc_length+3

            #최대 토큰을 넘는지 확인
            if total_source_len>=AiConfig.max_source_length:
                return None

            #truncating
            code_tokens=code_tokens[:AiConfig.max_source_length-3]
            source_tokens =[tokenizer.cls_token]+p_desc_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
            source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
            dfg=dfg[:AiConfig.max_source_length-len(source_tokens)]
            source_tokens+=[x[0] for x in dfg]
            position_idx+=[0 for x in dfg]
            source_ids+=[tokenizer.unk_token_id for x in dfg]
            padding_length=AiConfig.max_source_length-len(source_ids)
            position_idx+=[tokenizer.pad_token_id]*padding_length
            source_ids+=[tokenizer.pad_token_id]*padding_length
            source_mask = [1] * (len(source_tokens))
            source_mask+=[0]*padding_length

            #reindex
            reverse_index={}
            for idx,x in enumerate(dfg):
                reverse_index[x[1]]=idx
            for idx,x in enumerate(dfg):
                dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
            dfg_to_dfg=[x[-1] for x in dfg]
            dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
            length=len([tokenizer.cls_token])
            dfg_to_code=[(x[0]+length+p_desc_length,x[1]+length+p_desc_length) for x in dfg_to_code]

            target_tokens = tokenizer.tokenize("None")
            target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            target_mask = [1] *len(target_ids)
            padding_length = AiConfig.max_target_length - len(target_ids)
            target_ids+=[tokenizer.pad_token_id]*padding_length
            target_mask+=[0]*padding_length

            features.append(
                self.InputFeatures(
                    example_index,
                    source_ids,
                    position_idx,
                    dfg_to_code,
                    dfg_to_dfg,
                    target_ids,
                    source_mask,
                    target_mask,
                )
            )
        return features
    
    class TextDataset(Dataset):
        """
        모델 입력 형식에 맞도록 데이터셋을 텐서로 변환
        """
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, item):
            #calculate graph-guided masked function
            attn_mask=np.zeros((AiConfig.max_source_length,AiConfig.max_source_length),dtype=bool)
            #calculate begin index of node and max length of input
            node_index=sum([i>1 for i in self.examples[item].position_idx])
            max_length=sum([i!=1 for i in self.examples[item].position_idx])
            #sequence can attend to sequence
            attn_mask[:node_index,:node_index]=True
            #special tokens attend to all tokens
            for idx,i in enumerate(self.examples[item].source_ids):
                if i in [0,2]:
                    attn_mask[idx,:max_length]=True
            #nodes attend to code tokens that are identified from
            for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
                if a<node_index and b<node_index:
                    attn_mask[idx+node_index,a:b]=True
                    attn_mask[a:b,idx+node_index]=True
            #nodes attend to adjacent nodes
            for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
                for a in nodes:
                    if a+node_index<len(self.examples[item].position_idx):
                        attn_mask[idx+node_index,a+node_index]=True

            return (torch.tensor(self.examples[item].source_ids),
                    torch.tensor(self.examples[item].source_mask),
                    torch.tensor(self.examples[item].position_idx),
                    torch.tensor(attn_mask),
                    torch.tensor(self.examples[item].target_ids),
                    torch.tensor(self.examples[item].target_mask),)

    def remove_multi_line_comment(self,old_code):
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
    ## 2023.7 코드 추상화를 위한 함수 추가
    def abstract_pl(self,code,ident):
        """
        raw code를 추상화
        """
        tree = self.tree_parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        self.method,self.var=0,0

        def dfs(node):
            if node.child_count==0:
                if node.type=='identifier':
                    if node.parent.type not in ["for_in_clause","call","attribute","dotted_name","except_clause"]:
                        if (not ident.get(node.text)) and (node.text not in [b'i',b'j',b'k',b'__name__',b'int',b'str',b'input',b'float']):
                            if node.parent.type=="function_definition":
                                ident[(node.text)]="method"+str(self.method)
                                self.method+=1
                            else:
                                ident[(node.text)]="var"+str(self.var)
                                self.var+=1
                return
            else:
                for i in range(node.child_count):
                    dfs(node.children[i])
        dfs(root_node)
        for i in ident.items():
            code=re.sub(r'\b'+i[0].decode()+r'\b',i[1] , code)

        return code,ident
    
    def process(self,code,p_id):
        """
        모델 추론
        p_id를 풀이하는 raw code중 틀린 곳을 고친 코드를 리턴

        params
        - code : raw code
        - p_id : 문제 번호 ex) "p00001"
        -------------------------------------------
        return
        - 틀린 곳을 고친 코드
        - 원래의 버그가 있는 코드
        """
        p_desc_filename=p_id+".txt"
        if p_desc_filename not in os.listdir(self.folder_path+"/desc"):
            raise Exception(p_desc_filename+" 문제 설명문이 존재하지 않습니다.")
        #AST로 변환 후 code로 재변환으로 코드 일반화, tab 공백("    ")을 이스케이프문자 \t로 변환
        node=ast.parse(code)
        generalized_code=ast.unparse(node).replace("    ","\t")
        #주석 제거
        generalized_code=self.remove_multi_line_comment(generalized_code)
        #코드 변수, 함수 이름 추상화
        ident_list={}
        abstracted_code,ident_list=self.abstract_pl(generalized_code,ident_list)

        eval_examples = [self.Example(
                            source=abstracted_code,
                            target="",
                            p_id=p_id
                            )]
        eval_features = self.convert_examples_to_features(eval_examples, self.tokenizer)
        if eval_features is None:#최대 토큰 512를 넘으면 종료
            return None
        eval_data = self.TextDataset(eval_features)

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=64,num_workers=4)

        self.model.eval()
        p=[]
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch
            with torch.no_grad():
                preds = self.model(source_ids,source_mask,position_idx,att_mask)
                for pred in preds:
                    t=pred[0].cpu().numpy()
                    t=list(t)
                    if 0 in t:
                        t=t[:t.index(0)]
                    text = self.tokenizer.decode(t,clean_up_tokenization_spaces=False)
                    for index in ident_list.items():
                        text=text.replace(index[1],str(index[0],'utf-8'))
                    p.append(text)
        print(ident_list)
        print("-------------------------------")
        print(p[0])
        print("-------------------------------")
        print(generalized_code)
        return p[0],generalized_code
    
wpc=WPC()
