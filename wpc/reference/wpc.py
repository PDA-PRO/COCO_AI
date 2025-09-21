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
Inference-optimized version: remove DataLoader; run direct tensorized inference.
"""

from __future__ import absolute_import
import ast
import re
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import os, tempfile, shutil, torch
import random
import logging
from io import open
import torch.nn as nn
import torch._inductor.config as inductor_cfg
import torch._dynamo as dynamo
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from .parser.DFG import DFG_python
from .parser.utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token)
from .config import AiConfig
from .model import Seq2Seq



class WPC():
    def __init__(self, use_cuda: bool = True, use_dataparallel: bool = False):
        # 로거 설정
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 문제 설명 토큰 캐시
        self._desc_cache = {}

        # 폴더경로 추출
        self.folder_path = '/'.join(__file__.split("/")[:-2])

        # data flow graph 생성을 위한 tree_parser
        PY_LANGUAGE = Language(tspython.language())
        self.tree_parser = Parser(PY_LANGUAGE)

        # 코드 추상화에 필요한 변수
        self.method = 0
        self.var = 0

        # Device
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")

        # Seed (추론에서는 재현성보다 성능 권장)
        random.seed(42)
        torch.manual_seed(42)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")  # TF32 활성화(Ampere+)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        self.load_model_from_pt()

        if use_dataparallel and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        inductor_cfg.max_autotune = False
        dynamo.config.cache_size_limit = 64

        self.model.decode_step = torch.compile(
            self.model.decode_step,
            mode="reduce-overhead",     # 초기 지연 낮음
            fullgraph=False,            # 그래프 브레이크 허용
            dynamic=True                # 길이 가변 대응
        )

        # 버킷 목록( decode_step의 _bucket_len과 일치 )
        L_BUCKETS = (32, 64, 128, 256, 512)
        S_BUCKETS = (64, 128, 256, 512)
        
        with torch.inference_mode():
            K, H = AiConfig.beam_size, self.model.config.hidden_size
            for Lb in L_BUCKETS:
                for Sb in S_BUCKETS:
                    input_ids   = torch.zeros((K, Lb), dtype=torch.long, device=self.device)
                    context     = torch.zeros((Sb, K, H), dtype=torch.float32, device=self.device)
                    context_mask= torch.ones((K, Sb),   dtype=torch.long,  device=self.device)  # ‘유효=1’
                    _ = self.model.decode_step(input_ids, context, context_mask)

    
    def _write_file_bytes(self, path: str, data: bytes):
        with open(path, "wb") as f:
            f.write(data)

    def load_model(self):
        # 베이스 모델/토크나이저 설정
        config_class, tokenizer_class = RobertaConfig, RobertaTokenizer
        config = config_class.from_pretrained("microsoft/graphcodebert-base")
        config.add_pooling_layer = False
        self.tokenizer = tokenizer_class.from_pretrained("microsoft/graphcodebert-base")

        # Build model
        encoder = RobertaModel.from_pretrained(
            "microsoft/graphcodebert-base",
            config=config
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            config=config,
            beam_size=AiConfig.beam_size,
            max_length=AiConfig.max_length,
            sos_id=self.tokenizer.cls_token_id,
            eos_id=self.tokenizer.sep_token_id
        )

        # 가중치 로드
        model_path = os.path.join(self.folder_path, "pytorch_model.bin")
        if self.device.type == "cuda":
            self.model.load_state_dict(torch.load(model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        self.model.to(self.device).eval()


    def load_model_from_pt(self):
        """
        단일 .pt 번들에서 tokenizer/Seq2Seq/model meta를 복원
        """
        pt_path = os.path.join(self.folder_path, "wpc_bundle.pt")
        assert os.path.exists(pt_path), f"{pt_path} no exists"
        obj = torch.load(pt_path)
        assert obj.get("format") == "wpc_bundle_v1", "unknown bundle format"

        # 1) config 복원
        cfg = RobertaConfig.from_dict(obj["config"])
        # pooler 경고 방지
        cfg.add_pooling_layer = False

        # 2) 토크나이저 파일을 임시 폴더에 풀고 from_pretrained
        tmp = tempfile.mkdtemp(prefix="wpc_tok_load_")
        try:
            for fn, data in obj["tokenizer_files"].items():
                self._write_file_bytes(os.path.join(tmp, fn), data)
            self.tokenizer = RobertaTokenizer.from_pretrained(tmp)

            # 3) 모델 골격 생성 → state_dict 주입
            encoder = RobertaModel(cfg)  # from_pretrained 대신 config로 빈 모형 생성
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=cfg.hidden_size,
                nhead=cfg.num_attention_heads,
            )
            decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

            self.model = Seq2Seq(
                encoder=encoder,
                decoder=decoder,
                config=cfg,
                beam_size=AiConfig.beam_size,
                max_length=AiConfig.max_length,
                sos_id=self.tokenizer.cls_token_id,
                eos_id=self.tokenizer.sep_token_id,
            )
            self.model.load_state_dict(obj["state_dict"], strict=False)
            self.model.to(self.device).eval()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def extract_dataflow(self, code):
        """
        remove comments, tokenize code and extract dataflow
        """
        # remove comments
        try:
            code = remove_comments_and_docstrings(code)
        except Exception:
            pass
        # obtain dataflow
        try:
            tree = self.tree_parser.parse(bytes(code, 'utf8'))
            root_node = tree.root_node
            tokens_index = tree_to_token_index(root_node)
            code = code.split('\n')
            code_tokens = [index_to_code_token(x, code) for x in tokens_index]
            index_to_code = {}
            for idx, (index, code_tok) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, code_tok)
            try:
                DFG, _ = DFG_python(root_node, index_to_code, {})
            except Exception:
                DFG = []
            DFG = sorted(DFG, key=lambda x: x[1])
            indexs = set()
            for d in DFG:
                if len(d[-1]) != 0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_DFG = []
            for d in DFG:
                if d[1] in indexs:
                    new_DFG.append(d)
            dfg = new_DFG
        except Exception:
            dfg = []
        return code_tokens, dfg

    class InputFeatures(object):
        """A single training/test features for an example."""
        def __init__(self,
                     source_ids,
                     position_idx,
                     dfg_to_code,
                     dfg_to_dfg,
                     source_mask,
                     ):
            self.source_ids = source_ids
            self.position_idx = position_idx
            self.dfg_to_code = dfg_to_code
            self.dfg_to_dfg = dfg_to_dfg
            self.source_mask = source_mask

    # 내부: 문제 설명 토큰 캐시
    def _get_desc_tokens(self, p_id, tokenizer):
        cached = self._desc_cache.get(p_id)
        if cached is not None:
            return cached
        path = os.path.join(self.folder_path, "desc", p_id + ".txt")
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        toks = tokenizer.tokenize(txt)
        self._desc_cache[p_id] = toks
        return toks

    ## 2023.7 모델 입력 임베딩으로 문제 설명 추가
    def convert_to_feature(self, code, p_id):
        """
        모델에 입력값으로 쓸 수 있도록 임베딩
        - 길이 체크: 예제별(curr_len) 기준으로 수행
        - 문제 설명 토큰화 캐시 사용
        """

        # extract data flow
        code_tokens, dfg = self.extract_dataflow(code)
        code_tokens = [self.tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else self.tokenizer.tokenize(x)
                        for idx, x in enumerate(code_tokens)]
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        code_tokens = [y for x in code_tokens for y in x]

        # problem description (cached)
        p_desc_tokens = self._get_desc_tokens(p_id, self.tokenizer)
        p_desc_length = len(p_desc_tokens) + 1  # + [SEP]

        # 예제별 길이 검사
        curr_len = len(code_tokens) + (p_desc_length) + 3  # [CLS], [SEP], [SEP]
        if curr_len >= AiConfig.max_length:
            return None  # 넘치면 일단 종료 (기존 동작 유지)

        # truncating
        code_tokens = code_tokens[:AiConfig.max_length - 3]
        source_tokens = [self.tokenizer.cls_token] + p_desc_tokens + [self.tokenizer.sep_token] + code_tokens + [self.tokenizer.sep_token]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i + self.tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg = dfg[:AiConfig.max_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for _ in dfg]
        source_ids += [self.tokenizer.unk_token_id for _ in dfg]
        padding_length = AiConfig.max_length - len(source_ids)
        position_idx += [self.tokenizer.pad_token_id] * padding_length
        source_ids += [self.tokenizer.pad_token_id] * padding_length
        source_mask = [1] * (len(source_tokens)) + [0] * padding_length

        # reindex
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([self.tokenizer.cls_token])
        dfg_to_code = [(x[0] + length + p_desc_length, x[1] + length + p_desc_length) for x in dfg_to_code]

        return self.InputFeatures(
                    source_ids,
                    position_idx,
                    dfg_to_code,
                    dfg_to_dfg,
                    source_mask,
                )

    # 내부: feature -> attn_mask (TextDataset.__getitem__ 로직 이식)
    def _build_attn_mask_from_feature(self, feat):
        L = AiConfig.max_length
        attn_mask = torch.zeros((L, L), dtype=torch.bool, device=self.device)

        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in feat.position_idx])
        max_length = sum([i != 1 for i in feat.position_idx])

        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True

        # special tokens attend to all tokens
        for idx, tok_id in enumerate(feat.source_ids):
            if tok_id in [0, 2]:  # [CLS]=0, [SEP]=2 for RoBERTa tokenizer ids here
                attn_mask[idx, :max_length] = True

        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(feat.dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True

        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(feat.dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(feat.position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return attn_mask

    # 내부: feature -> 배치 텐서 준비
    def _prepare_batch_from_feature(self, feat):
        to = self.device
        source_ids = torch.as_tensor(feat.source_ids, dtype=torch.long, device=to).unsqueeze(0)
        source_mask = torch.as_tensor(feat.source_mask, dtype=torch.long, device=to).unsqueeze(0)
        position_idx = torch.as_tensor(feat.position_idx, dtype=torch.long, device=to).unsqueeze(0)
        attn_mask = self._build_attn_mask_from_feature(feat).unsqueeze(0)
        return source_ids, source_mask, position_idx, attn_mask

    def remove_multi_line_comment(self, old_code):
        p = re.compile('(\"\"\")')
        swt = True
        word_range = []
        for i in p.finditer(old_code):
            if swt:
                word_range.append(i.span()[0])
                swt = False
            else:
                word_range.append(i.span()[1])
                swt = True
        if len(word_range):
            new_code = ""
            new_code += old_code[:word_range[0]]
            for i in range(1, len(word_range) - 1, 2):
                new_code += old_code[word_range[i]:word_range[i + 1]]
            new_code += old_code[word_range[-1]:]
            return new_code
        else:
            return old_code

    ## 2023.7 코드 추상화를 위한 함수 추가
    def abstract_pl(self, code, ident):
        """
        raw code를 추상화
        """
        tree = self.tree_parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        self.method, self.var = 0, 0

        def dfs(node):
            if node.child_count == 0:
                if node.type == 'identifier':
                    if node.parent.type not in ["for_in_clause", "call", "attribute", "dotted_name", "except_clause"]:
                        if (not ident.get(node.text)) and (node.text not in [b'i', b'j', b'k', b'__name__', b'int', b'str', b'input', b'float']):
                            if node.parent.type == "function_definition":
                                ident[(node.text)] = "method" + str(self.method)
                                self.method += 1
                            else:
                                ident[(node.text)] = "var" + str(self.var)
                                self.var += 1
                return
            else:
                for i in range(node.child_count):
                    dfs(node.children[i])
        dfs(root_node)
        for i in ident.items():
            code = re.sub(r'\b' + i[0].decode() + r'\b', i[1], code)

        return code, ident

    def process(self, code, p_id):
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
        p_desc_filename = p_id + ".txt"
        if p_desc_filename not in os.listdir(os.path.join(self.folder_path, "desc")):
            raise Exception(p_desc_filename + " 문제 설명문이 존재하지 않습니다.")

        # AST → 코드 재생성(일반화) + 탭 정규화
        node = ast.parse(code)
        generalized_code = ast.unparse(node).replace("    ", "\t")
        # 주석 제거
        generalized_code = self.remove_multi_line_comment(generalized_code)
        # 변수/함수명 추상화
        ident_list = {}
        abstracted_code, ident_list = self.abstract_pl(generalized_code, ident_list)


        feature = self.convert_to_feature(abstracted_code, p_id)

        if feature is None:  # 최대 토큰 초과 시 종료
            return None

        source_ids, source_mask, position_idx, attn_mask = self._prepare_batch_from_feature(feature)

        # 추론
        self.model.eval()
        with torch.inference_mode():
            if self.device.type == "cuda":
                major = torch.cuda.get_device_properties(0).major
                amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    preds = self.model(source_ids, source_mask, position_idx, attn_mask)
            else:
                preds = self.model(source_ids, source_mask, position_idx, attn_mask)
        # 후처리: top-1 beam을 디코딩
        pred_beams = preds[0]              # [beam_size, max_len]
        top1 = pred_beams[0].detach().cpu().tolist()
        if 0 in top1:
            top1 = top1[:top1.index(0)]
        text = self.tokenizer.decode(top1, clean_up_tokenization_spaces=False)
        # 식별자 복원
        for k, v in ident_list.items():
            text = text.replace(v, str(k, 'utf-8'))

        return text, generalized_code
    
wpc = WPC()
