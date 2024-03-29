# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn.functional as F
from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens


@register_strategy('nomask_predict')
class NoMaskPredict(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations
        # self.use_at = args.use_at
        # self.use_at_iter = args.use_at_iter
    
    def generate(self, model, encoder_out, tgt_tokens, tgt_dict):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        iterations = seq_len if self.iterations is None else self.iterations
        
        tgt_tokens, token_probs = self.generate_non_autoregressive(model, encoder_out, tgt_tokens)
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())  # pad 部分转换为 pad()
        assign_single_value_byte(token_probs, pad_mask, 1.0)   # pad 部分概率变为1， 以便后续计算整个sentence的概率
        #print("Initialization: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        
        for counter in range(1, iterations):
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()

            assign_single_value_byte(token_probs, pad_mask, 1.0)
            # mask_ind = self.select_worst(token_probs, num_mask)
            mask_ind = self.select_repeat(tgt_tokens, token_probs, num_mask)
            assign_single_value_long(tgt_tokens, mask_ind, tgt_dict.mask())
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())

            shift_tgt_tokens = tgt_tokens[:, :-1]
            shift_tgt_tokens = torch.cat(
                [tgt_tokens.new(shift_tgt_tokens.size(0), 1).fill_(tgt_dict.bos()), shift_tgt_tokens],dim=1)

            #print("Step: ", counter+1)
            #print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[0]))
            # if self.use_at and counter > self.use_at_iter:
            #     gen_decoder_out = model.g_decoder(shift_tgt_tokens, encoder_out, self_attn=True)
            # else:
            gen_decoder_out = model.g_decoder(shift_tgt_tokens, encoder_out, self_attn=False)
            gen_dec_logits = F.linear(gen_decoder_out[0], model.decoder_embed_tokens.weight)
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(gen_dec_logits)
            
            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            
            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            # tgt_tokens = new_tgt_tokens
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            #print("Prediction: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        
        lprobs = token_probs.log().sum(-1)  # 这里计算的是 log_prob 之和
        return tgt_tokens, lprobs
    
    def generate_non_autoregressive(self, model, encoder_out, tgt_tokens):
        gen_decoder_out = model.g_decoder(tgt_tokens, encoder_out, self_attn=False)
        # x, {'attn': attn, 'inner_states': inner_states, 'predicted_lengths': encoder_out['predicted_lengths']} 
        # print(decoder_out[0].shape)  # [batch, max_len, decoder_emb_dim]
        
        gen_dec_logits = F.linear(gen_decoder_out[0], model.decoder_embed_tokens.weight)
        # print(gen_dec_logits.shape)  # [batch, max_len, decoder_outuput_dim]

        tgt_tokens, token_probs, _ = generate_step_with_prob(gen_dec_logits)
        return tgt_tokens, token_probs

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)

    def select_repeat(self, tgt_tokens, token_probs, num_mask):
        bsz, seq_len = tgt_tokens.size()
        # masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = []
        for batch in range(bsz):
            mask_pos = []
            for pos in range(1, seq_len):
                if tgt_tokens[batch, pos] == tgt_tokens[batch, pos - 1]:
                    mask_pos.append(pos)
                    mask_pos = torch.LongTensor(mask_pos).type_as(tgt_tokens)
                if len(mask_pos) == 0:
                    mask_pos = token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1]
            masks.append(mask_pos)
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return masks

