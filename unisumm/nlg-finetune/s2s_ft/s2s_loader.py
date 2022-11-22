import numpy as np

from random import randint, shuffle, choice
from random import random as rand
import math
import logging
import torch
import torch.utils.data


logger = logging.getLogger(__name__)


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def _get_word_split_index(tokens, st, end):
    split_idx = []
    i = st
    while i < end:
        if (not tokens[i].startswith('##')) or (i == st):
            split_idx.append(i)
        i += 1
    split_idx.append(end)
    return split_idx


def _expand_whole_word(tokens, st, end):
    new_st, new_end = st, end
    while (new_st >= 0) and tokens[new_st].startswith('##'):
        new_st -= 1
    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
        new_end += 1
    return new_st, new_end


class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.skipgram_prb = None
        self.skipgram_size = None
        self.pre_whole_word = None
        self.mask_whole_word = None
        self.word_subsample_prb = None
        self.sp_prob = None
        self.pieces_dir = None
        self.vocab_words = None
        self.pieces_threshold = 10
        self.call_count = 0
        self.offline_mode = False
        self.skipgram_size_geo_list = None
        self.span_same_mask = False

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, 
                 mode="s2s", pos_shift=False, source_type_id=0, target_type_id=1,
                 cls_token='[CLS]', sep_token='[SEP]', pad_token='[PAD]'):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))
        self.task_idx = 3   # relax projection layer for different tasks
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.max_tgt_length = max_tgt_length
        self.pos_shift = pos_shift


        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token

        self.source_type_id = source_type_id
        self.target_type_id = target_type_id

        self.cc = 0

    def __call__(self, instance):
        tokens, max_a_len = instance

        if max_a_len  > len(tokens):
            tokens += [self.pad_token] * \
                (max_a_len - len(tokens))

        if(tokens[-1]!=self.sep_token):
            tokens[-1] = self.sep_token

        input_ids = self.indexer(tokens)


        self.cc += 1
        if self.cc < 20:
            # print("Vocab size = %d" % len(self.vocab_words))
            # for tk_id in input_ids:
            #     print(u"trans %d -> %s" % (tk_id, self.vocab_words[tk_id]))
            logger.info(u"Input src = %s" % " ".join((self.vocab_words[tk_id]) for tk_id in input_ids))

        return input_ids
