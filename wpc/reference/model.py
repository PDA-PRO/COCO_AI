# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    """
        Build Sequence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """
    def __init__(self, encoder, decoder, config,
                 beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        # --- Replace 2048x2048 bias with compact precomputed causal mask ---
        # RoBERTa position embeddings limit is typically 514: clamp by config and max_length
        cfg_L = getattr(config, "max_position_embeddings", 514)
        L = min(max_length if max_length is not None else cfg_L, cfg_L)
        tgt_mask_base_bool = torch.triu(torch.ones((L, L), dtype=torch.bool), diagonal=1)
        self.register_buffer("tgt_mask_base", tgt_mask_base_bool)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending whether we are using TorchScript or not """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def _bucket_len(self, L: int, buckets=(32, 64, 128, 256, 512)) -> int:
        """입력 길이 L을 가장 가까운 상위 버킷으로 올림."""
        for b in buckets:
            if L <= b:
                return b
        return buckets[-1]

    def tie_weights(self):
        """ Share the input and output embeddings. """
        self._tie_or_clone_weights(self.lm_head, self.encoder.embeddings.word_embeddings)

    def forward(self, source_ids, source_mask, position_idx, attn_mask):
        # --- Build encoder inputs (node aggregation) ---
        nodes_mask = position_idx.eq(0)      # [B, N]
        token_mask = position_idx.ge(2)      # [B, T]
        inputs_embeddings = self.encoder.embeddings.word_embeddings(source_ids)  # [B, T, D]

        # [B, N, T] boolean mask -> normalize to weights and average token embeddings into node positions
        nodes_to_token_mask = (nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask)  # [B, N, T]
        denom = nodes_to_token_mask.sum(-1, keepdim=True).clamp_min(1e-10).float()          # [B, N, 1]
        w = nodes_to_token_mask.float() / denom                                             # [B, N, T]
        # einsum("abc,acd->abd") == bmm([B,N,T] x [B,T,D]) -> [B,N,D]
        avg_embeddings = torch.bmm(w, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        # --- Encoder forward ---
        outputs = self.encoder(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx
        )
        # [B, S, H] -> [S, B, H] using transpose (cheaper than permute+contiguous)
        encoder_output = outputs[0].transpose(0, 1)

        # --- Beam search decode ---
        preds = []
        device = source_ids.device
        zero = torch.zeros(1, dtype=torch.long, device=device)

        B = source_ids.shape[0]
        for i in range(B):
            # Context per sample
            context = encoder_output[:, i:i+1]           # [S, 1, H]
            context_mask_i = source_mask[i:i+1, :]       # [1, S]
            # Expand context to beam without allocating new memory
            context = context.expand(-1, self.beam_size, -1)           # [S, K, H]
            context_mask = context_mask_i.expand(self.beam_size, -1)   # [K, S]

            S = context.size(0)
            Sb = self._bucket_len(S, buckets=(64, 128, 256, 512))    # 필요에 맞게 조정
            if S < Sb:
                # context 패딩: 아래로 0 패딩
                pad_ctx = torch.zeros((Sb - S, self.beam_size, self.config.hidden_size),
                                    dtype=context.dtype, device=context.device)
                context = torch.cat([context, pad_ctx], dim=0)       # [Sb, K, H]

                # context_mask 패딩: 추가 구간은 '패딩(무효)' → 0 채움
                pad_m = torch.zeros((self.beam_size, Sb - S),
                                    dtype=context_mask.dtype, device=context_mask.device)
                context_mask = torch.cat([context_mask, pad_m], dim=1)  # [K, Sb]
            else:
                # 혹시 S가 버킷보다 크면 자르는 것도 가능(보통 max_source_length로 이미 제한됨)
                context = context[:Sb]
                context_mask = context_mask[:, :Sb]

            beam = Beam(self.beam_size, self.sos_id, self.eos_id, device=device)
            input_ids = beam.getCurrentState()           # [K, 1]

            for _ in range(self.max_length):
                if beam.done():
                    break

                log_probs = self.decode_step(input_ids, context, context_mask)

                beam.advance(log_probs)
                # Safely select origins (no .data)
                input_ids = input_ids.index_select(0, beam.getCurrentOrigin())  # [K, L]
                # Append new tokens
                input_ids = torch.cat((input_ids, beam.getCurrentState()), dim=-1)  # [K, L+1]

            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [
                torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1)
                for p in pred
            ]
            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)
        return preds
    
    def decode_step(self, input_ids, context, context_mask):
        """
        한 스텝 디코딩의 수치연산만 수행하여 log_probs를 반환.
        - 버킷 패딩으로 L 종류를 제한해 Inductor CUDAGraph 오버헤드 감소
        input_ids: [K, L] (현재까지 생성된 토큰)
        context:   [S, K, H]
        context_mask: [K, S]
        반환: [K, V]
        """
        K, L = input_ids.size(0), input_ids.size(1)
        Lb = self._bucket_len(L)                       # 버킷 길이 선택 (예: 32/64/128/256/512)

        # 1) tgt 마스크: [Lb, Lb] (사전계산 버퍼에서 슬라이스)
        tgt_mask = self.tgt_mask_base[:Lb, :Lb]          # dtype=bool

        # 2) input_ids를 오른쪽 패딩하여 [K, Lb]로 맞춤 (pad_token_id 사용 권장)
        if L < Lb:
            pad_id = getattr(self.config, "pad_token_id", self.eos_id)
            pad = input_ids.new_full((K, Lb - L), pad_id)
            input_ids_padded = torch.cat([input_ids, pad], dim=1)
        else:
            input_ids_padded = input_ids

        # 3) tgt_key_padding_mask: 패딩 위치(True=mask)
        #   - 디코더는 이 마스크로 패딩 토큰에 대한 어텐션/업데이트를 막음
        if L < Lb:
            false_part = torch.zeros((K, L), dtype=torch.bool, device=input_ids.device)
            true_part  = torch.ones((K, Lb - L), dtype=torch.bool, device=input_ids.device)
            tgt_kpm = torch.cat([false_part, true_part], dim=1)  # [K, Lb]
        else:
            tgt_kpm = torch.zeros((K, Lb), dtype=torch.bool, device=input_ids.device)

        # 4) 임베딩 → 디코더 호출
        tgt_embeddings = self.encoder.embeddings(input_ids_padded).transpose(0, 1)  # [Lb, K, D]
        out = self.decoder(
            tgt_embeddings,
            context,
            tgt_mask=tgt_mask,                              # bool
            tgt_key_padding_mask=tgt_kpm,                   # bool
            memory_key_padding_mask=(1 - context_mask).bool()
        )


        out = torch.tanh(self.dense(out))                    # [Lb, K, H]

        # 5) 진짜 마지막 위치 L-1의 히든만 사용 (패딩 구간은 버림)
        hidden_states = out[L - 1]                           # [K, H]
        log_probs = self.lsm(self.lm_head(hidden_states))    # [K, V]
        return log_probs

class Beam(object):
    def __init__(self, size, sos, eos, device):
        self.size = size
        self.device = device
        # The score for each translation on the beam.
        self.scores = torch.zeros(size, dtype=torch.float, device=device)
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [torch.full((size,), 0, dtype=torch.long, device=device)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1].view(-1, 1)

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk`:
        Compute and update the beam search.

        Parameters:
        * `wordLk`- log-probs of advancing from the last step (K x words)
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i].item() == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i].item() == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0].item() == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0].item() if torch.is_tensor(a[0]) else -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i].item() != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0].item() if torch.is_tensor(a[0]) else -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                # k는 tensor 스칼라일 수 있으므로 파이썬 int로 변환
                idx = k.item() if torch.is_tensor(k) else int(k)
                tok = self.nextYs[j+1][idx]
                hyp.append(tok)
                # 다음 스텝의 backpointer도 동일하게 int 인덱싱
                k = self.prevKs[j][idx]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok.item() == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
