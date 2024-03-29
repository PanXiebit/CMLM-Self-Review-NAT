# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
label_smoothed_length_gan_cross_entropy
"""

import math
import torch
import torch.nn as nn
from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('label_smoothed_length_gan_2_cross_entropy')
class LabelSmoothedLengthGan_2_CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.gen_weights = args.gen_weights
        self.dis_weights = args.dis_weights
        self.neg_weights = args.neg_weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bce_loss = nn.BCELoss(reduction="none")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--gen_weights', default=1., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--dis_weights', default=5., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--neg_weights', default=0.2, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--pos_weights', default=1.0, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])  
        # 这里的输出是长度为6的 list.
        # gen_dec_logits, dis_dec_logits, encoder_out['predicted_lengths'], fake_data, gen_decoder_out[1]["attn"], dis_decoder_out[1]["attn"]
        # print(net_output[0].shape)
        # print(net_output[1].shape)
        # print(net_output[2].shape)
        # print(net_output[3].shape)
        loss, nll_loss, length_loss, dis_loss, ntokens = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = ntokens  #TODO why not merge ntokens and sample_size? what is the difference?
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'length_loss': utils.item(length_loss.data) if reduce else length_loss.data,
            'dis_loss': utils.item(dis_loss.data) if reduce else dis_loss.data,
            'ntokens': ntokens,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # generator loss
        lprobs = model.get_normalized_probs(net_output, log_probs=True)  # logits = net_output[0], 计算其对应的 log_softmax
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)  # sample['target']
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        
        # length loss
        length_lprobs = net_output[2]   # predicted_lengths [batch, max_target_positions]
        length_target = sample['net_input']['prev_output_tokens'].ne(self.padding_idx).sum(-1).unsqueeze(-1)  # [batch, 1]
        #TODO doesn't work for dynamic length. change to eos-based method.
        length_loss = -length_lprobs.gather(dim=-1, index=length_target)
        
        # discriminator loss
#         dis_label_1 = net_output[3].eq(sample['net_input']['real_target']).type(torch.FloatTensor).to(self.device)  # [batch, tgt_len]
#         dis_loss_1 = self.bce_loss(torch.sigmoid(net_output[1].squeeze()), dis_label_1)
#         dis_loss_1 = dis_loss_1.view(-1, 1)[non_pad_mask]  # [batch_size, tgt_len]
#         print("\n dis_label_1", dis_label_1.view(-1, 1)[non_pad_mask].sum())
#         print("\ndis_loss_1", dis_loss_1.sum())
        
        fake_data = net_output[3]
        real_target = sample['net_input']['real_target']
        bs, tgt_len = fake_data.size()
        fake_data = fake_data.unsqueeze(-1).repeat([1, 1, tgt_len])   # [batch, tgt_len, tgt_len]
        real_target = real_target.unsqueeze(1).repeat([1,tgt_len,1])  # [batch, tgt_len, tgt_len]
        pos_weights = torch.eye(tgt_len,tgt_len).unsqueeze(0).repeat([bs, 1, 1]).to(self.device)
        neg_weights = (1- pos_weights).to(self.device)
        dis_label = fake_data.eq(real_target) # * weights.to(self.device) # [batch, tgt_len, tgt_len]
        pos_label = (dis_label * pos_weights).sum(-1)   # [batch, tgt_len]
        neg_label = (dis_label * neg_weights).sum(-1)
        
        pos_loss = self.bce_loss(torch.sigmoid(net_output[1].squeeze()), pos_label)    # [batch, tgt_len]
        neg_loss = -torch.log(1- torch.sigmoid(net_output[1].squeeze())) * neg_label.to(self.device)
        dis_loss = (pos_loss + neg_loss).view(-1, 1)[non_pad_mask]
        
        #print("\n dis_label", pos_label.view(-1, 1)[non_pad_mask].sum())
        #print("\ndis_loss\n", dis_loss.sum(), pos_loss.view(-1, 1)[non_pad_mask].sum())
        
        
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            length_loss = length_loss.sum()
            dis_loss = dis_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = self.gen_weights * ((1. - self.eps) * nll_loss + eps_i * smooth_loss) + length_loss + self.dis_weights * dis_loss
        ntokens = non_pad_mask.sum().data.item()
#         print(((1. - self.eps) * nll_loss + eps_i * smooth_loss)/ ntokens)
#         print(length_loss / ntokens)
#         print(self.dis_weights * dis_loss / ntokens)
#         print(loss / ntokens / math.log(2))
        return loss, nll_loss, length_loss, dis_loss, non_pad_mask.sum().data.item()

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'length_loss': sum(log.get('length_loss', 0) for log in logging_outputs) / nsentences / math.log(2),
            'dis_loss': sum(log.get('dis_loss', 0) for log in logging_outputs) / ntokens / math.log(2), 
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
