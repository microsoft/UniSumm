from __future__ import absolute_import, division, print_function

import logging
import math
import os
import json
import random
import glob
from functools import lru_cache
from typing import Union

import torch
import tqdm
import array
import collections
import torch.utils.data
from transformers import RobertaTokenizer, BartTokenizer, AddedToken
from transformers.file_utils import WEIGHTS_NAME
import torch.nn.functional as F

try:
    import lmdb
except:
    pass

OPTIM_NAME = "optimizer.bin"

# TASK_MAP = {"pubmed": ['taskword%03d' % i for i in range(64)],
#             "arxiv": ['taskword%03d' % i for i in range(64, 128)],
#             "billsum": ['taskword%03d' % i for i in range(128, 192)],
#             "gov": ['taskword%03d' % i for i in range(192, 256)], "none": '<s>'}

logger = logging.getLogger(__name__)


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(torch.autograd.Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax
        Returns
        -------
        output : torch.Tensor
            same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold
        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax
        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor
        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size



def _get_invalid_locations_mask_fixed_dilation(seq_len: int, w: int, d: int):
    diagonals_list = []
    for j in range(-d * w, d, d):
        diagonal_mask = torch.zeros(seq_len, device='cpu', dtype=torch.uint8)
        diagonal_mask[:-j] = 1
        diagonals_list.append(diagonal_mask)
    return torch.stack(diagonals_list, dim=-1)

@lru_cache()
def _get_invalid_locations_mask(w: int, d: Union[torch.Tensor,int], autoregressive: bool, device: str):
    if isinstance(d, int):
        affected_seq_len = w * d
        mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
        mask = mask[None, :, None, :]
    else:
        affected_seq_len = w * d.max()
        head_masks = []
        d_list = d.cpu().numpy().tolist()
        for d in d_list:
            one_head_mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
            head_masks.append(one_head_mask)
        mask = torch.stack(head_masks, dim=-2)
        mask = mask[None, :, :, :]

    ending_mask = None if autoregressive else mask.flip(dims=(1, 3)).bool().to(device)
    return affected_seq_len, mask.bool().to(device), ending_mask


class SparsemaxFunc(torch.autograd.Function):
    """Sparsemax function."""

    @staticmethod
    def forward(ctx, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax


        device=input.device

        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape

        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        dim = 1
        output = ctx.saved_tensors[0]

        nonzeros = torch.ne(output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim, keepdim=True) / torch.sum(nonzeros, dim=dim, keepdim=True)
        grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return grad_input


class Seq2seqDatasetForBert(torch.utils.data.Dataset):
    def __init__(
            self, features, max_source_len, max_target_len,
            vocab_size, cls_id, sep_id, pad_id, offset, num_training_instances,  num_max_mask_token=0,
            ):
        if isinstance(features, list):
            self.features = features
        else:
            self.features = DocDB(features)
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.offset = offset
        if offset > 0:
            logger.info("  ****  Set offset %d in Seq2seqDatasetForBert ****  ", offset)
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.num_training_instances = num_training_instances
        self.num_max_mask_token = num_max_mask_token

    def __len__(self):
        return self.num_training_instances

    def __trunk(self, ids, max_len):
        if len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def __pad(self, ids, max_len):
        if len(ids) < max_len:
            return ids + [self.pad_id] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids

    def __getitem__(self, _idx):
        idx = (self.offset + _idx) % len(self.features)
        # print("%d get %d" % (_idx, idx))
        feature = self.features[idx]
        prefix_ids = feature["prefix_ids"]
        source_ids = self.__trunk(feature["source_ids"], self.max_source_len)
        target_ids = feature["target_ids"]
        target_ids = self.__trunk(target_ids, self.max_target_len)
        if(source_ids[-1]!=self.sep_id):
            source_ids[-1] = self.sep_id
        if(target_ids[-1]!=self.sep_id):
            target_ids[-1] = self.sep_id

        num_source_tokens = len(source_ids)
        num_target_tokens = len(target_ids)-1

        labels = target_ids[1:]
        target_ids = target_ids[:-1]

        source_ids = self.__pad(source_ids, self.max_source_len)
        target_ids = self.__pad(target_ids, self.max_target_len)
        labels = self.__pad(labels, self.max_target_len)

        return source_ids, target_ids, labels, prefix_ids, num_source_tokens, num_target_tokens


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "ckpt-*/%s" % WEIGHTS_NAME))
    fn_optim_list = glob.glob(os.path.join(output_dir, "ckpt-*/%s" % OPTIM_NAME))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(os.path.dirname(fn).split('-')[-1]) for fn in fn_model_list]
                   ) & set([int(os.path.dirname(fn).split('-')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def get_checkpoint_state_dict(output_dir, ckpt):
    model_recover_checkpoint = os.path.join(output_dir, "ckpt-%d" % ckpt, WEIGHTS_NAME)
    logger.info(" ** Recover model checkpoint in %s ** ", model_recover_checkpoint)
    model_state_dict = torch.load(model_recover_checkpoint, map_location='cpu')
    optimizer_recover_checkpoint = os.path.join(output_dir, "ckpt-%d" % ckpt, OPTIM_NAME)
    checkpoint_state_dict = torch.load(optimizer_recover_checkpoint, map_location='cpu')
    checkpoint_state_dict['model'] = model_state_dict
    return checkpoint_state_dict


def report_length(length_counter, total_count):
    max_len = max(length_counter.keys())
    a = 0
    tc = 0
    while a < max_len:
        cc = 0
        for i in range(16):
            cc += length_counter[a + i]

        tc += cc
        if cc > 0:
            logger.info("%d ~ %d = %d, %.2f%%" % (a, a + 16, cc, (tc * 100.0) / total_count))
        a += 16


def serialize_str(x):
    return u"{}".format(x).encode('ascii')


def serialize_array(x, dtype):
    data = array.array(dtype)
    data.fromlist(x)
    return data.tobytes()

def write_to_lmdb(db, key, value):
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key, value)
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit*2
            print('>>> Doubling LMDB map size to %sMB ...' %
                  (new_limit >> 20,))
            db.set_mapsize(new_limit)  # double it


def deserialize_str(x):
    return x.decode('ascii')


class DocDB(object):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.start_key_index = int(deserialize_str(txn.get(b'__start__')))
            self.size = int(deserialize_str(txn.get(b'__size__')))
            self.dtype = deserialize_str(txn.get(b'__dtype__'))

    def _deserialize_array(self, x):
        data = array.array(self.dtype)
        data.frombytes(x)
        return data.tolist()

    def __getitem__(self, doc_id):
        with self.env.begin(write=False) as txn:
            example = {
                "source_ids": self._deserialize_array(txn.get(b"src_ids_%d" % doc_id)), 
                "target_ids": self._deserialize_array(txn.get(b"tgt_ids_%d" % doc_id)), 
            }
        return example

    def __len__(self):
        return self.size

def save_shards(features, cached_features_dir, num_shards):
    os.makedirs(cached_features_dir, exist_ok=True)
    shard_size = math.floor(len(features) / num_shards)
    for i in range(num_shards):
        shard_features = features[i * shard_size: (i + 1) * shard_size]
        cached_shard_file = os.path.join(cached_features_dir, "shard_{}.pt".format(i))
        torch.save(shard_features, cached_shard_file)
        
def load_shards(cached_features_dir):
    listing = glob.glob(os.path.join(cached_features_dir, "shard_*.pt"))
    num_shards = len(listing)
    features = []
    for i in range(num_shards):
        cache_fp = os.path.join(cached_features_dir, "shard_{}.pt".format(i))
        shard_features = torch.load(cache_fp)
        features.extend(shard_features)
    return features



def load_and_cache_examples(
        example_files, tokenizer, local_rank, cached_features_dir, task_map_file, use_universal_prefix=-1, shuffle=True, eval_mode=False, num_shards=1,  decode_task='', new_task=None, tasks='',num_prefixs_per_task=0):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    use_universal_prefix = int(use_universal_prefix)
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()



    example_files = example_files.split(',')
    if(decode_task!=''):
        tasks = [decode_task]
    else:
        if (tasks == ''):
            tasks = ['none'] * len(example_files)
        else:
            tasks = tasks.split(',')


    if(task_map_file is not None and os.path.exists(task_map_file)):
        TASK_MAP = json.load(open(task_map_file))
    else:
        TASK_MAP = {}
        for i, t in enumerate(tasks):
            TASK_MAP[t] = [i for i in range(i * num_prefixs_per_task, (i + 1) * num_prefixs_per_task)]

    if(new_task):
        i = len(TASK_MAP)
        TASK_MAP[new_task] =  [i for i in range(i * num_prefixs_per_task, (i + 1) * num_prefixs_per_task)]
        tasks =  [new_task]


    if cached_features_dir is not None and os.path.isdir(cached_features_dir):
        logger.info("Loading features from cached dir %s", cached_features_dir)
        features = load_shards(cached_features_dir)
    else:
        if use_universal_prefix<0:
            features = []

            for task, example_file in zip(tasks, example_files):
                examples = []
                logger.info("Creating features from dataset file at %s", example_file)

                with open(example_file, mode="r", encoding="utf-8") as reader:
                    for line in reader:
                        examples.append(json.loads(line))

                slc = collections.defaultdict(int)
                tlc = collections.defaultdict(int)

                for example in tqdm.tqdm(examples):

                    if isinstance(example["src"], list):
                        source_tokens = example["src"]
                        target_tokens = [] if eval_mode else example["tgt"]
                    else:
                        source_tokens = tokenizer.tokenize(example["src"])
                        target_tokens = [] if eval_mode else tokenizer.tokenize(example["tgt"])

                    source_ids = tokenizer.convert_tokens_to_ids(source_tokens[1:])
                    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)




                    prefix_ids = TASK_MAP[task]


                    slc[len(source_ids)] += 1
                    tlc[len(target_ids)] += 1

                    features.append({
                            "prefix_ids": prefix_ids,
                            "source_ids": source_ids,
                            "target_ids": target_ids,
                        })

            if shuffle:
                random.shuffle(features)
                logger.info("Shuffle the features !")

            logger.info("Source length:")
            report_length(slc, total_count=len(examples))
            logger.info("Target length:")
            report_length(tlc, total_count=len(examples))

            if local_rank in [-1, 0] and cached_features_dir is not None:
                logger.info("Saving features into cached dir %s", cached_features_dir)
                # torch.save(features, cached_features_file)
                save_shards(features, cached_features_dir, num_shards=num_shards)
                task_map_file = os.path.join(cached_features_dir, 'task_map.json')
                logger.info("Saving TASK_MAP at %s", task_map_file)
                with open(task_map_file, 'w') as f:
                    json.dump(TASK_MAP, f)
        else:
            features = []
            # print(tasks)
            if 'universal' in tasks:
                non_universal_tasks = tasks[:]
                non_universal_tasks.remove('universal')
            else:
                non_universal_tasks = tasks[:]         # random_features = []
            # print(example_files)
            if len(non_universal_tasks)!=len(example_files):
                print("WARNING!! Task number does not equal file number!")
            for task, example_file in zip(non_universal_tasks, example_files):
                examples = []
                
                logger.info("Creating features from dataset file at %s", example_file)
                logger.info("Using universal prefix")
                with open(example_file, mode="r", encoding="utf-8") as reader:
                    for line in reader:
                        examples.append(json.loads(line))

                example_probs = torch.rand(len(examples))

                slc = collections.defaultdict(int)
                tlc = collections.defaultdict(int)

                for example_prob, example in tqdm.tqdm(zip(example_probs, examples)):

                    if isinstance(example["src"], list):
                        source_tokens = example["src"]
                        target_tokens = [] if eval_mode else example["tgt"]
                    else:
                        source_tokens = tokenizer.tokenize(example["src"])
                        target_tokens = [] if eval_mode else tokenizer.tokenize(example["tgt"])

                    source_ids = tokenizer.convert_tokens_to_ids(source_tokens[1:])
                    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
                    slc[len(source_ids)] += 1
                    tlc[len(target_ids)] += 1
                    if example_prob>0.15:
                        prefix_ids = TASK_MAP[task]
                        features.append({
                                "prefix_ids": prefix_ids,
                                "source_ids": source_ids,
                                "target_ids": target_ids,
                            })
                    else:
                        prefix_ids = TASK_MAP['universal']
                        features.append({
                                "prefix_ids": prefix_ids,
                                "source_ids": source_ids,
                                "target_ids": target_ids,
                            })

            if shuffle:
                random.shuffle(features)
                logger.info("Shuffle the features !")

            logger.info("Source length:")
            report_length(slc, total_count=len(examples))
            logger.info("Target length:")
            report_length(tlc, total_count=len(examples))

            if local_rank in [-1, 0] and cached_features_dir is not None:
                logger.info("Saving features into cached dir %s", cached_features_dir)
                # torch.save(features, cached_features_file)
                save_shards(features, cached_features_dir, num_shards=num_shards)
                task_map_file = os.path.join(cached_features_dir, 'task_map.json')
                logger.info("Saving TASK_MAP at %s", task_map_file)
                with open(task_map_file, 'w') as f:
                    json.dump(TASK_MAP, f)

    # if local_rank in [-1, 0] and (task_map_file is None or task_map_file=='') :
    #     logger.info("Saving TASK_MAP at %s", task_map_file)
    #
    #     with open(os.path.join(cached_features_dir, 'task_map.json'), 'w') as f:
    #         json.dump(TASK_MAP, f)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features


# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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


