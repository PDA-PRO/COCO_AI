# bundle_pack.py
import os, torch, tempfile, shutil
from transformers import RobertaConfig, RobertaTokenizer
from typing import Dict, Any

def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def save_all_to_single_pt(
    model,                    # Seq2Seq (encoder+decoder 포함)
    tokenizer: RobertaTokenizer,
    config: RobertaConfig,
    out_path: str,
    meta: Dict[str, Any] = None,   # beam_size, max_length 등 런타임 메타
    hf_id: str = "microsoft/graphcodebert-base",
):
    """
    model.state_dict + config + tokenizer 파일들 + meta 를 단일 .pt로 직렬화
    """
    # 1) 토크나이저를 임시 폴더에 표준 형태로 저장
    tmp = tempfile.mkdtemp(prefix="wpc_tok_")
    try:
        tokenizer.save_pretrained(tmp)         # vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json 등
        # 2) 파일 바이트 모아서 딕셔너리로 구성
        tok_files = {}
        for fn in os.listdir(tmp):
            full = os.path.join(tmp, fn)
            if os.path.isfile(full):
                tok_files[fn] = _read_file_bytes(full)

        # 3) 최종 payload 구성
        payload = {
            "format": "wpc_bundle_v1",
            "hf_id": hf_id,
            "config": config.to_dict(),            # HF config dict
            "tokenizer_files": tok_files,          # 토크나이저 파일들(바이트)
            "state_dict": model.state_dict(),      # Seq2Seq 전체 가중치
            "meta": meta or {},                    # 예: {"beam_size": 4, "max_length": 256}
        }

        # 4) 저장 (zip 기반 컨테이너로 직렬화됨)
        torch.save(payload, out_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)