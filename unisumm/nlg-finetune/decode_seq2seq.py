"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import glob
import logging
import argparse
import math
import re

from torch import nn
from tqdm import tqdm
import numpy as np
import torch
import random

from s2s_ft.modeling_bart import BartForConditionalGeneration
import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.utils import load_and_cache_examples
from transformers import \
    BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer, BartConfig, BartTokenizer

from s2s_ft.my_tokenization_bart import BartTokenizer

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'roberta': RobertaTokenizer,
    'xlm-roberta': XLMRobertaTokenizer,
}



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path to config.json for the model.")
    parser.add_argument("--decode_task", default=None, type=str, required=True)

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--no_cuda', action='store_true',
                        help="Whether to use CUDA for decoding")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_ngrams', default=-1, type=int)

    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--sep_enc_dec', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--nwindow', type=int, default=8)
    parser.add_argument("--enc_layer_types", type=str, default='WWWWWWWWWWWW')
    parser.add_argument("--dec_layer_types", type=str, default='WWWWWWWWWWWW')

    parser.add_argument('--no_rel', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--task_map_file',  type=str,
                        help="task_map_file.")

    args = parser.parse_args()

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    else:
        random_seed = random.randint(0, 10000)
        logger.info("Set random seed as: {}".format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None)


    vocab = tokenizer.encoder

    if hasattr(tokenizer, 'model_max_length'):
        tokenizer.model_max_length = args.max_seq_length
    elif hasattr(tokenizer, 'max_len'):
        tokenizer.max_len = args.max_seq_length

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token])

#    print(args.model_path)
    found_checkpoint_flag = False
    for model_recover_path in glob.glob(args.model_path):
        if not os.path.isdir(model_recover_path):
            continue

        logger.info("***** Recover model: %s *****", model_recover_path)

        config_file = args.config_path if args.config_path else os.path.join(model_recover_path, "config.json")
        logger.info("Read decoding config from: %s" % config_file)
        config = BartConfig.from_json_file(config_file)

        bi_uni_pipeline = []
        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
            list(vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
            max_tgt_length=args.max_tgt_length,
            cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token))

        found_checkpoint_flag = True

        model = BartForConditionalGeneration(config)

        model.add_prefix_module(config)

        # model.model.encoder.embed_prefix = nn.Embedding(config.num_prefixs, config.hidden_size)
        # model.model.encoder.prefix_mlp_key = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.prefix_dim),
        #     nn.Tanh(),
        #     nn.Linear(config.prefix_dim, config.encoder_layers * config.hidden_size))
        # model.model.encoder.prefix_mlp_value = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.prefix_dim),
        #     nn.Tanh(),
        #     nn.Linear(config.prefix_dim, config.encoder_layers * config.hidden_size))


        state_dict = torch.load(os.path.join(model_recover_path,'pytorch_model.bin'), map_location='cpu')
        model.load_state_dict(state_dict, strict=False  )
        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length


        TASK_MAP = json.load(open(args.task_map_file))
        task_tokens = TASK_MAP[args.decode_task]

        to_pred = load_and_cache_examples(
            args.input_file, tokenizer, local_rank=-1,
            cached_features_dir=None, shuffle=False, eval_mode=True, decode_task=args.decode_task, task_map_file=args.task_map_file)


        input_lines = []
        for line in to_pred:
            input_lines.append(tokenizer.convert_ids_to_tokens(line["source_ids"])[:max_src_length])
        if args.subset > 0:
            logger.info("Decoding subset: %d", args.subset)
            input_lines = input_lines[:args.subset]

        input_lines = sorted(list(enumerate(input_lines)),
                             key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        with tqdm(total=total_batch) as pbar:
            batch_count = 0
            first_batch = True
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += args.batch_size
                batch_count += 1
                # max_a_len = max([len(x) for x in buf])
                max_a_len = args.max_seq_length
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                prefix_ids = [task_tokens]*len(instances)
                segment_ids = []
                with torch.no_grad():
                    input_ids = torch.tensor(instances, dtype=torch.long)
                    input_ids = input_ids.to(device)
                    prefix_ids = torch.tensor(prefix_ids, dtype=torch.long)
                    prefix_ids = prefix_ids.to(device)


                    summary_ids = model.generate(input_ids, prefix_ids,num_beams=args.beam_size, min_length=args.min_len, max_length=args.max_tgt_length, length_penalty=args.length_penalty,
                                                 no_repeat_ngram_size=args.forbid_ngrams, early_stopping=True, decoder_start_token_ids=[tokenizer.bos_token_id])
                    summaries =  [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                         summary_ids]
                    for i in range(len(buf)):
                        summary = summaries[i]
                        summary = summary.replace('\n',' ')
                        summary = re.sub(' +',' ', summary)
                        if first_batch or batch_count % 10 == 0:
                            logger.info("{} = {}".format(buf_id[i], summary))
                        output_lines[buf_id[i]] = summary
                pbar.update(1)
                first_batch = False
        if args.output_file:
            fn_out = args.output_file
        else:
            fn_out = model_recover_path + '.' + args.split
        with open(fn_out, "w", encoding="utf-8") as fout:
            for l in output_lines:
                fout.write(l)
                fout.write("\n")

    if not found_checkpoint_flag:
        logger.info("Not found the model checkpoint file!")


if __name__ == "__main__":
    main()
