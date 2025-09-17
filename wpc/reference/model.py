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
        # Upper-triangular (strict) is -inf; diagonal and below are 0.
        tgt_mask_base = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        self.register_buffer("tgt_mask_base", tgt_mask_base)

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

            beam = Beam(self.beam_size, self.sos_id, self.eos_id, device=device)
            input_ids = beam.getCurrentState()           # [K, 1]

            # Expand context to beam without allocating new memory
            context = context.expand(-1, self.beam_size, -1)           # [S, K, H]
            context_mask = context_mask_i.expand(self.beam_size, -1)   # [K, S]

            for _ in range(self.max_length):
                if beam.done():
                    break

                L_step = input_ids.size(1)
                tgt_mask = self.tgt_mask_base[:L_step, :L_step]        # [L, L], float

                # Decoder input embeddings: use encoder's embeddings (includes position embeddings)
                tgt_embeddings = self.encoder.embeddings(input_ids).transpose(0, 1)  # [L, K, D]

                out = self.decoder(
                    tgt_embeddings,
                    context,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=(1 - context_mask).bool()
                )
                out = torch.tanh(self.dense(out))       # [L, K, H]
                hidden_states = out[-1]                 # last step: [K, H]
                log_probs = self.lsm(self.lm_head(hidden_states))  # [K, V]

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
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
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
