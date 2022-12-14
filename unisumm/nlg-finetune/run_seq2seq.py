from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler

from s2s_ft.modeling_bart import BartForConditionalGeneration
from s2s_ft.my_tokenization_bart import BartTokenizer

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import tqdm

from transformers import AdamW, get_linear_schedule_with_warmup, BartConfig
from transformers import \
    RobertaConfig, BertConfig, \
    BertTokenizer, RobertaTokenizer, \
    XLMRobertaConfig, XLMRobertaTokenizer

from s2s_ft import utils

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'roberta': (RobertaConfig, RobertaTokenizer),
    'xlm-roberta': (XLMRobertaConfig, XLMRobertaTokenizer),
    'bart': (BartConfig, BartTokenizer),
}


def prepare_for_training(args, model, checkpoint_state_dict, amp):
    no_decay = ['bias', 'LayerNorm.weight']
    prefix_decay = ['prefix']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and (not any(nd in n for nd in prefix_decay))],
         'weight_decay': args.weight_decay_lm},
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and (any(nd in n for nd in prefix_decay))],
         'weight_decay': args.weight_decay_prefix},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if checkpoint_state_dict:
        optimizer.load_state_dict(checkpoint_state_dict['optimizer'])
        model.load_state_dict(checkpoint_state_dict['model'])

        # then remove optimizer state to make amp happy
        # https://github.com/NVIDIA/apex/issues/480#issuecomment-587154020
        if amp:
            optimizer.state = {}

    if amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        if checkpoint_state_dict:
            amp.load_state_dict(checkpoint_state_dict['amp'])

            # Black Tech from https://github.com/NVIDIA/apex/issues/480#issuecomment-587154020
            # forward, backward, optimizer step, zero_grad
            random_input = {'input_ids': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'decoder_input_ids': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'labels': torch.ones(size=(2, 2), device=args.device, dtype=torch.long),
                            'num_source_tokens': torch.zeros(size=(2,), device=args.device, dtype=torch.long),
                            'num_target_tokens': torch.zeros(size=(2,), device=args.device, dtype=torch.long)}

            loss = model(**random_input)
            print("Loss = %f" % loss.cpu().item())
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            model.zero_grad()

            # then load optimizer state_dict again (this time without removing optimizer.state)
            optimizer.load_state_dict(checkpoint_state_dict['optimizer'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    return model, optimizer


def train(args, training_features, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0] and args.log_dir:
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        tb_writer = None

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    # model recover
    recover_step = utils.get_max_epoch_model(args.output_dir)

    if recover_step:
        checkpoint_state_dict = utils.get_checkpoint_state_dict(args.output_dir, recover_step)
    else:
        checkpoint_state_dict = None

    model.to(args.device)
    model, optimizer = prepare_for_training(args, model, checkpoint_state_dict, amp=amp)

    per_node_train_batch_size = args.per_gpu_train_batch_size * args.n_gpu * args.gradient_accumulation_steps
    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    global_step = recover_step if recover_step else 0

    if args.num_training_steps == -1:
        args.num_training_steps = args.num_training_epochs * len(training_features) / train_batch_size

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps, last_epoch=-1)

    if checkpoint_state_dict:
        scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])

    train_dataset = utils.Seq2seqDatasetForBert(
        features=training_features, max_source_len=args.max_source_seq_length,
        max_target_len=args.max_target_seq_length, vocab_size=tokenizer.vocab_size,
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
        offset=train_batch_size * global_step, num_training_instances=train_batch_size * args.num_training_steps,
        num_max_mask_token=args.num_max_mask_token,
    )

    logger.info("Check dataset:")
    for i in range(5):
        source_ids, target_ids, labels, prefix_ids = train_dataset.__getitem__(i)[:4]
        logger.info("Instance-%d" % i)
        logger.info("Source tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(source_ids)))
        logger.info("Target tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(target_ids)))
        logger.info("Label tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(labels)))
        logger.info("prefix_ids = %s" % str(prefix_ids))

#    logger.info("Mode = %s" % str(model))

    # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", len(training_features))
    logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_features))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Batch size per node = %d", per_node_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.num_training_steps)

    if args.num_training_steps <= global_step:
        logger.info("Training is done. Please use a new dir or clean this dir!")
    else:
        # The training features are shuffled
        train_sampler = SequentialSampler(train_dataset) \
            if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=per_node_train_batch_size // args.gradient_accumulation_steps,
            collate_fn=utils.batch_list_to_batch_tensors)

        train_iterator = tqdm.tqdm(
            train_dataloader, initial=global_step * args.gradient_accumulation_steps,
            desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=args.local_rank not in [-1, 0])

        model.train()
        model.zero_grad()

        tr_loss, logging_loss = 0.0, 0.0

        for step, batch in enumerate(train_iterator):
            if global_step > args.num_training_steps:
                break
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'decoder_input_ids': batch[1],
                      'labels': batch[2],
                      'prefix_ids': batch[3],
                      'num_source_tokens': batch[4],
                      'num_target_tokens': batch[5]}
            model_outputs = model(**inputs)

            loss = model_outputs['loss']

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            logging_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("")
                    logger.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
                    logging_loss = 0.0

                if (torch.distributed.is_initialized()):
                    self_rank = torch.distributed.get_rank()
                else:
                    self_rank = args.local_rank

                if self_rank in [-1, 0] and args.save_steps > 0 and \
                        (global_step % args.save_steps == 0 or global_step == args.num_training_steps):
                    save_path = os.path.join(args.output_dir, "ckpt-%d" % global_step)
                    os.makedirs(save_path, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(save_path)

                    # optim_to_save = {
                    #     "optimizer": optimizer.state_dict(),
                    #     "lr_scheduler": scheduler.state_dict(),
                    # }
                    # if args.fp16:
                    #     optim_to_save["amp"] = amp.state_dict()
                    # torch.save(optim_to_save, os.path.join(save_path, utils.OPTIM_NAME))

                    logger.info("Saving model checkpoint %d into %s", global_step, save_path)

    if args.local_rank in [-1, 0] and tb_writer:
        tb_writer.close()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--train_source_file", default=None, type=str, required=True,
    #                     help="Training data contains source")
    # parser.add_argument("--train_target_file", default=None, type=str, required=True,
    #                     help="Training data contains target")
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--tasks", default='', type=str, required=True,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--num_shards", default=1, type=int,
                        help="The output directory where the log will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_source_seq_length", default=464, type=int,
                        help="The maximum total source sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_target_seq_length", default=48, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--cached_train_features_dir", default=None, type=str,
                        help="Cached training features file")
    parser.add_argument("--task_map_file", default=None, type=str,
                        help="task_map_file")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay_lm", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay_prefix", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_training_steps", default=-1, type=int,
                        help="set total number of training steps to perform")
    parser.add_argument("--num_training_epochs", default=10, type=int,
                        help="set total number of training epochs to perform (--num_training_steps has higher priority)")
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--random_prob", default=0.1, type=float,
                        help="prob to random replace a masked token")
    parser.add_argument("--keep_prob", default=0.1, type=float,
                        help="prob to keep no change for a masked token")
    parser.add_argument("--fix_word_embedding", action='store_true',
                        help="Set word embedding no grad when finetuning.")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument('--source_mask_prob', type=float, default=-1.0,
                        help="Probability to mask source sequence in fine-tuning")
    parser.add_argument('--target_mask_prob', type=float, default=-1.0,
                        help="Probability to mask target sequence in fine-tuning")
    parser.add_argument('--num_max_mask_token', type=int, default=0,
                        help="The number of the max masked tokens in target sequence")
    parser.add_argument('--mask_way', type=str, default='v2',
                        help="Fine-tuning method (v0: position shift, v1: masked LM, v2: pseudo-masking)")
    parser.add_argument("--lmdb_cache", action='store_true',
                        help="Use LMDB to cache training features")
    parser.add_argument("--lmdb_dtype", type=str, default='h',
                        help="Data type for cached data type for LMDB")


    parser.add_argument("--enc_layer_types", type=str, default='FFFFFFFFFFFF')


    parser.add_argument("--window_size", type=int, default=512)


    parser.add_argument("--prefix_dim", type=int, default=768)

    parser.add_argument("--sparsemax", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--num_prefixs_per_task", type=int, nargs='?',
                        const=True, default=64,
                        help="num_prefixs_per_task.")

    parser.add_argument("--only_train_prefix", type=int, nargs='?',
                        const=True, default=-1,
                        help="only_train_prefix.")
    parser.add_argument("--only_train_prefix_pos", type=int, nargs='?',
                        const=True, default=-1,
                        help="only_train_prefix_pos.")

    parser.add_argument("--use_prefix", type=int, nargs='?',
                        const=True, default=-1,
                        help="use_prefix.")
    parser.add_argument("--use_universal_prefix", type=int, nargs='?',
                        const=True, default=-1,
                        help="use_universal_prefix.")

    parser.add_argument("--only_train_prefix_emb", type=int, nargs='?',
                        const=True, default=-1,
                        help="only_train_prefix_emb.")

    parser.add_argument("--new_task", type=str, nargs='?',
                        const=True, default=None,
                        help="new_task.")


    parser.add_argument("--load_from", type=str, nargs='?',
                        const=True, default=None,
                        help="load_from.")

    args = parser.parse_args()
    return args


def init_distributed_itp(args):
    args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    os.environ['LOCAL_RANK'] = str(args.gpu)
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def prepare(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        if args.no_cuda:
            device = torch.device("cpu")
            args.n_gpu = 1

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def get_model_and_tokenizer(args):
    config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    # config = BertForSeq2SeqConfig.from_exist_config(
    #     config=model_config, label_smoothing=args.label_smoothing,
    #     fix_word_embedding=args.fix_word_embedding,
    #     max_position_embeddings=args.max_source_seq_length + args.max_target_seq_length)

    config.max_src_len = args.max_source_seq_length
    config.max_tgt_len = args.max_target_seq_length
    config.label_smoothing = args.label_smoothing

    config.enc_layer_types = args.enc_layer_types
    config.window_size = args.window_size

    tasks = args.tasks.split(',')

    num_prefixs = args.num_prefixs_per_task * len(tasks)

    config.num_prefixs = num_prefixs

    config.num_prefixs_per_task = args.num_prefixs_per_task

    config.prefix_dim = args.prefix_dim

    config.use_prefix = args.use_prefix
    config.use_universal_prefix = args.use_universal_prefix

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None)

    # model = BartForConditionalGeneration(config)


    if(args.load_from):
        logger.info("***** Recover model: %s *****", args.load_from)
        config_file = os.path.join(args.load_from, "config.json")
        logger.info("Read decoding config from: %s" % config_file)
        config = BartConfig.from_json_file(config_file)
        config.use_prefix = args.use_prefix
        config.use_universal_prefix = args.use_universal_prefix
        state_dict = torch.load(os.path.join(args.load_from,'pytorch_model.bin'), map_location='cpu')
        model = BartForConditionalGeneration(config)
        if (config.use_prefix > 0):
            model.add_prefix_module(config)

        model.load_state_dict(state_dict, strict=False)
        if(config.use_prefix > 0 and args.new_task):
            model.add_new_task(config, num_prefixs_per_task = args.num_prefixs_per_task)

    else:
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model.adjust_model(config)
        if (config.use_prefix > 0):
            model.add_prefix_module(config)

    model.add_label_smoothing(config)
    logger.info("Model config for seq2seq: %s", str(config))

    if (args.only_train_prefix > 0):
        for name, param in model.named_parameters():
            param.requires_grad = False
            if (args.only_train_prefix_emb > 0):
                if (name.find("embed_prefix") >= 0):
                    param.requires_grad = True
            elif (name.find("prefix") >= 0):
                param.requires_grad = True

    if (args.only_train_prefix_pos > 0):
        for name, param in model.named_parameters():
            param.requires_grad = False
            if (args.only_train_prefix_emb > 0):
                if (name.find("embed_prefix") >= 0):
                    param.requires_grad = True
            elif (name.find("prefix") >= 0):
                param.requires_grad = True
            if (name.find("embed_pos") >= 0):
                param.requires_grad = True

    return model, tokenizer


def main():
    args = get_args()
    prepare(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab
    # Load pretrained model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    if args.cached_train_features_dir is None:
        args.cached_train_features_dir = os.path.join(args.output_dir, "cached_features_for_training")

    training_features = utils.load_and_cache_examples(
        example_files=args.train_file, tokenizer=tokenizer, local_rank=args.local_rank,
        cached_features_dir=args.cached_train_features_dir, task_map_file=args.task_map_file, shuffle=True,
        num_shards=args.num_shards, new_task=args.new_task, tasks=args.tasks,
        use_universal_prefix=args.use_universal_prefix,
        num_prefixs_per_task=args.num_prefixs_per_task)

    train(args, training_features, model, tokenizer)


if __name__ == "__main__":
    main()
