#WA 코드쌍 추출
from __future__ import absolute_import
import argparse
import ast
import re
from tree_sitter import Language, Parser
import os
import torch
import random
import logging
import numpy as np
from io import open
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
from .parser import DFG_python
from .parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token)
logger = logging.getLogger(__name__)

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
tree_parser = Parser()
tree_parser.set_language(PY_LANGUAGE)

method=0
var=0

#remove comments, tokenize code and extract dataflow     
def extract_dataflow(code, lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = tree_parser.parse(bytes(code,'utf8'))    
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
        

def convert_examples_to_features(examples, tokenizer, args,p_desc_path,stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples,total=len(examples))):
        ##extract data flow
        code_tokens,dfg=extract_dataflow(example.source,'python')
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        
        #problem description
        p_desc_tokens=[]
        with open(os.path.join(p_desc_path,example.p_id+".txt"),"r") as p_desc_file:
            p_desc_tokens=tokenizer.tokenize(p_desc_file.read())
        p_desc_length=len(p_desc_tokens)+1

        #truncating
        code_tokens=code_tokens[:args.max_source_length-3]
        source_tokens =[tokenizer.cls_token]+p_desc_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg=dfg[:args.max_source_length-len(source_tokens)]
        source_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        source_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args.max_source_length-len(source_ids)
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

        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                logger.info("position_idx: {}".format(position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, dfg_to_dfg))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
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
    def __init__(self, examples, args):
        self.examples = examples
        self.args=args  
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.max_source_length,self.args.max_source_length),dtype=bool)
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
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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
    tree = tree_parser.parse(bytes(code, "utf8"))
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--code", default=None, type=str, required=True,
                        help="런타임 오류는 없지만 테스트 케이스를 틀린 코드")
    parser.add_argument("--p_id", default=None, type=str, required=True,
                        help="문제 id")
    parser.add_argument("--p_desc_path", default=None, type=str, required=True,
                        help="Path to problem descriptions" ) 
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--load_model_path", default=None, type=str, required=True,
                        help="Path to trained model: Should contain the .bin files" )   
    args = parser.parse_args()
    #AST로 변환 후 code로 재변환으로 코드 일반화, tab 공백("    ")을 이스케이프문자 \t로 변환
    node=ast.parse(args.code)
    generalized_code=ast.unparse(node).replace("    ","\t")
    #주석 제거
    generalized_code=remove_multi_line_comment(generalized_code)

    #코드 변수, 함수 이름 추상화
    ident_list={}
    abstracted_code,ident_list=abstract_pl(generalized_code,ident_list)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Set seed
    set_seed(42)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
    config = config_class.from_pretrained("microsoft/graphcodebert-base")
    tokenizer = tokenizer_class.from_pretrained("microsoft/graphcodebert-base")
    
    #budild model
    encoder = model_class.from_pretrained("microsoft/graphcodebert-base",config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=256,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path),strict=False)
        
    model.to(device)
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    eval_examples = [Example(
                        source=abstracted_code,
                        target="",
                        p_id=args.p_id
                        )]
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,args.p_desc_path,stage='test')
    eval_data = TextDataset(eval_features,args) 

    # Calculate bleu
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

    model.eval() 
    p=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch                    
        with torch.no_grad():
            preds = model(source_ids,source_mask,position_idx,att_mask)  
            for pred in preds:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                for index in ident_list.items():
                    text.replace(index[1],index[0])
                p.append(text)
    print(p)
    print(args.code)
            
if __name__ == "__main__":
    main()