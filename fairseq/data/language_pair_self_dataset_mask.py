# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
import random

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, bos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, add_bos=True
):
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False, add_bos=add_bos, is_decoder_src=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(data_utils.collate_tokens(
                    [s[key][i] for s in samples], pad_idx, eos_idx, bos_idx, left_pad=False, 
                    add_bos=add_bos, is_decoder_src=is_decoder_src
                ))
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples], pad_idx, eos_idx, bos_idx, left_pad=False, 
                add_bos=add_bos, is_decoder_src=is_decoder_src
            )

    is_target_list = isinstance(samples[0]['dec_target'], list)
    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'ntokens': sum(s['ntokens'] for s in samples),
        'net_input': {
            'src_tokens': merge('enc_source', add_bos=add_bos),
            'src_lengths': torch.LongTensor([
                s['enc_source'].numel() for s in samples
            ]),
            'prev_output_tokens': merge('dec_source', add_bos=add_bos, is_decoder_src=True),
            'real_target': merge('real_target', add_bos=False)
        },
        'target': merge('dec_target', is_target_list, add_bos=False),
        'nsentences': samples[0]['enc_source'].size(0),
    }

    """id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)
        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)
    
    batch = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        'nsentences': samples[0]['source'].size(0),
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch"""


class LanguagePairSelfDatasetMask(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.
    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side.
            Default: ``True``
        left_pad_target (bool, optional): pad target tensors on the left side.
            Default: ``False``
        max_source_positions (int, optional): max number of tokens in the source
            sentence. Default: ``1024``
        max_target_positions (int, optional): max number of tokens in the target
            sentence. Default: ``1024``
        shuffle (bool, optional): shuffle dataset elements before batching.
            Default: ``True``
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing.
            Default: ``True``
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None, 
        left_pad_source=True, left_pad_target=False, add_bos=True,
        max_source_positions=2048, max_target_positions=2048,
        shuffle=True, input_feeding=True,
        dynamic_length=False,
        mask_range=False,
        train=True,
        seed=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.dynamic_length = dynamic_length
        self.mask_range = mask_range
        self.train = train
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.seed = seed
        
        self.add_bos = add_bos


    def __getitem__(self, index):
        enc_source, dec_source, dec_target, real_target, ntokens = self._make_source_target(self.src[index], self.tgt[index])
        return {'id': index, 'enc_source': enc_source, 'dec_source': dec_source, 
                'dec_target': dec_target, 'real_target': real_target,'ntokens': ntokens}

    def __len__(self):
        return len(self.src)

    def _make_source_target(self, source, target):
        if self.dynamic_length:
            max_len = 3 * len(source) // 2 + 1
            target = target.new((target.tolist() + ([self.tgt_dict.eos()] * (max_len - len(target))))[:max_len])
        
        min_num_masks = 1
        
        enc_source = source
        dec_source = target.new(target.tolist())
        dec_target_cp = target.new(target.tolist())
        dec_target = target.new([self.tgt_dict.pad()] * len(dec_source))
        
        if self.train:
            if min_num_masks < len(dec_source):
                sample_size = self.random.randint(min_num_masks, len(dec_source))
            else:
                sample_size = len(dec_source)

            if self.mask_range:
                start = self.random.randint(len(dec_source) - sample_size + 1)
                ind = list(range(start, start + sample_size))
            else:
                ind = self.random.choice(len(dec_source) , size=sample_size, replace=False)
            
            dec_source[ind] = self.tgt_dict.mask()
            dec_target[ind] = dec_target_cp[ind]
        else:
            dec_target = dec_target_cp
            dec_source[:] = self.tgt_dict.mask()

        ntokens = dec_target.ne(self.tgt_dict.pad()).sum(-1).item()
        real_target = dec_target_cp
        #print ("masked tokens", self.tgt_dict.string(dec_source))
        #print ("original tokens", self.tgt_dict.string(dec_target))
        #print ("source tokens", self.src_dict.string(enc_source))

        return enc_source, dec_source, dec_target, real_target, ntokens

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(), bos_idx=self.src_dict.bos(), 
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, add_bos=self.add_bos
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = num_tokens // max(src_len, tgt_len)

        enc_source, dec_source, dec_target, ntokens = self._make_source_target(self.src_dict.dummy_sentence(src_len), self.tgt_dict.dummy_sentence(tgt_len))

        return self.collater([
            {
                'id': i,
                'enc_source': enc_source,
                'dec_source': dec_source,
                'dec_target': dec_target,
                'ntokens': ntokens,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle and self.train and self.seed is None:
            return np.random.permutation(len(self))
        
        indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.src, 'supports_prefetch')
            and self.src.supports_prefetch
            and hasattr(self.tgt, 'supports_prefetch')
            and self.tgt.supports_prefetch
        )
