# UniSumm
<p align="center">
    <img src="./figures/11.png" width="444" alt="method">
</p>


## 0. Overview
In this work, we introduce, **UniSumm** a unified model that can be easily prefix-tuned to excel at any few-shot summarization tasks and a new benchmark, **SummZoo** for diverse and robust few-shot summarization evaluation.

*[UniSumm: Unified Few-shot Summarization with Multi-Task Pre-Training and Prefix-Tuning](https://arxiv.org/abs/2211.09783)*

If you use UniSumm and SummZoo, please kindly cite our paper:
```
@article{chen2022unisumm,
  title={UniSumm: Unified Few-shot Summarization with Multi-Task Pre-Training and Prefix-Tuning},
  author={Chen, Yulong and Liu, Yang and Xu, Ruochen and Yang, Ziyi and Zhu, Chenguang and Zeng, Michael and Zhang, Yue},
  journal={arXiv preprint arXiv:2211.09783},
  year={2022}
}
```

### Content
1. [**UniSumm**](#1-unisumm)
2. [**SummZoo**](#summzoo)
3. [**Setup**](#setup)
4. [**Usage**](#usage)
5. [**Evaluation**](#evaluation)
6. [**Citation**](#citation)



## 1. UniSumm
<p align="center">
    <img src="./figures/unisumm.png" width="888" alt="method">
</p>

As shown in the above figure, the overall framework of UniSumm contains 2 phases: 1) learning summarization general knowledge by multi-task pre-training and; 2)learning target task knowledge by prefix-tuning.

### (a) Multi-task Pre-Training with Prefix
We take BART as the foundation model, and further pre-train it on a set of summarization datasets.
In particular, for each task $t$, we inject task-specific encoder and decoder prefx vectors $P^t = [P^t_{en};P^t_{de}]$ into the model, which are prepended to each Transformer layer.
In this stage, we optimize the parameters of BART and prefixes togeter.

In addition, we also pre-train a universal encoder and decoder prefix vectors, $P^*$.
For each training instance from task $t$ it has a 15\% probability to be coupled with this universal prefix vectors, instead of its original prefix $P^t$.
This universal prefix vector can be used as a robust initialization for few-shot prefix-tuning in the 2nd phase.

We release the multi-task pre-trained UniSumm checkpoint, which can be used for few-shot summarization.

### (b) Prefix-Tuning for Few-Shot Summarization
Through multi-task pre-training, the model is equipped with diverse summarization knowledge.
For an unseen task $u$, we create a new prefix vector $P^u$, which can be initialized using the above universal prefix $P^*$.
Then we freeze the parameter of summarization model and only update this new prefix vector.

## SummZoo
Previous work on few-shot summariztaion either focuses on one type of data, e.g., monologue, or experiments on miscellaneous few-shot samples.
These make the evaluation from different research not comparable.

Thus, we assemble a new benchmark, SummZoo, based on two principles, namely diversity and robustness.

<p align="center">
    <img src="./figures/summzoo.png" width="666" alt="summzoo">
</p>

### Data Collection
We assemble SummZoo from 8 popular summarization datasets.
The statistic is shown in the above table.

We tokenized data in SummZoo, which can be directly used for training UniSumm.
You can download the tokenized data [here](https://drive.google.com/file/d/1gjOqQ_GKJp2LZxUCiblGGqA60V7epQHB/view?usp=share_link), and untokenized data [here](https://drive.google.com/file/d/1_sTerMkKJVPSXVDDK-F-mzZ1fTvR4NB9/view?usp=share_link).


## Setup
For training and inference, please setup the following environment using Python>=3.7 (mind the cuda version suitable to your GPU):

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip clone https://github.com/microsoft/UniSumm.git
cd UniSumm/unisumm
pip install -r requirements.txt
```

To use ```fp16```, you need to install [apex](https://github.com/NVIDIA/apex) by running:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

And you can unzip the downloaded SummZoo data in the data folder (/UniSumm/unisumm/data): ```unzip Summzoo.zip -d data```.


## Usage
### Get the Multi-task Pre-trained UniSumm (Phase-1 Model)
We release the checkpoint of multi-task pre-trained UniSumm [here](https://drive.google.com/file/d/1dDF284w5FfzwcZ_08b070pdy4bdbc02a/view?usp=share_link).

After downloaded, unzip it in the current folder (UniSumm/unisumm): ```unzip unisumm_model.zip```.

### Prefix-tune the Phase-1 Model
Then, ```cd nlg-finetune```, and you can easily tune UniSumm using prefix-tuning. 
The following command shows an example of how to tune UniSumm on 10-shot QMSum (sample group 42) on 4 A100 GPUs.
```
export TRAIN_FILE=../data/Summzoo/qmsum/10-shot/train.jsonl.h10.s42
export OUTPUT_DIR=../few-shot-unisumm/qmsum/10-42
export CACHE_DIR=../cache/unisumm
export LOAD_FROM=../unisumm_model/ckpt-300000/
export TASK_MAP_FILE=../unisumm_model/task_map.json

rm $OUTPUT_DIR -rf

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --master_port 8888 --nproc_per_node 4 \
run_seq2seq.py \
--train_file $TRAIN_FILE \
--output_dir $OUTPUT_DIR  \
--model_type bart \
--model_name_or_path facebook/bart-large \
--fp16 \
--fp16_opt_level O2 \
--max_source_seq_length 2048 \
--max_target_seq_length 400 \
--per_gpu_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--learning_rate 1.5e-4 \
--num_warmup_steps 10 \
--num_training_steps 100 \
--cache_dir $CACHE_DIR \
--save_steps 100 \
--num_shards 1 \
--num_prefixs_per_task 256 \
--tasks 'universal' \
--logging_steps 50 \
--use_prefix 1 \
--only_train_prefix 1 \
--load_from $LOAD_FROM \
--task_map_file $TASK_MAP_FILE
```

Note that for fair comparison, the hyperparameter choice is based on previous work and empirical consideration.
In real practice, you could search hyperparameters for better performance.

### Inference
The following command shows an example of using prefix-tuned UniSumm to conduct inference on the QMSum testset:

```
export MODEL_PATH=../few-shot-unisumm/qmsum/10-42/ckpt-100
export SAVE_PATH=../unisumm_outs/qmsum.10.42
export TASK_MAP_FILE=../few-shot-unisumm/qmsum/10-42/cached_features_for_training/task_map.json
export INPUT_FILE=../data/Summzoo/qmsum/test/qmsum.test.bart.uncased.jsonl

mkdir ../unisumm_outs

CUDA_VISIBLE_DEVICES=0 python decode_seq2seq.py \
--fp16 \
--do_lower_case \
--model_path $MODEL_PATH \
--max_seq_length 2048 \
--max_tgt_length 256 \
--batch_size 4 \
--beam_size 5 \
--length_penalty 0.6 \
--mode s2s \
--min_len 60 \
--input_file $INPUT_FILE \
--decode_task 'universal' \
--task_map_file $TASK_MAP_FILE \
--output_file $SAVE_PATH
```

Then you should see the output file ```qmsum.10.42``` in the ```/UniSumm/unisumm/unisumm_outs``` folder.


### Get All Results in One Command
In the ```nlg-finetune``` folder, by running the below command, you will get models trained on all sub-tasks (5 sample sets in both 10/100 settings) in SummZoo, using 4 GPUs, and output files generated by those models.
```
bash run.sh
```
Note that the above command may not utilize your GPUs effiently and the inference time can be parcularly long, you need to modifiy it according to your resources.


## Evaluation
We use files2rouge for evaluation, you could refer to [files2rouge](https://github.com/pltrdy/files2rouge#getting-started) to setup and evaluate:
You should get similar scores by evaluating the above generated summary on QMSum testset:
```
Rouge-1 F1: 0.36482
Rouge-2 F1: 0.11488
Rouge-L F1: 0.23300
```

## Citation
When using SummZoo for evaluation, please cite all individual papers where orginial datasets are introduced. 
See [Summzoo.bib](https://github.com/microsoft/UniSumm/blob/main/Summzoo.bib)



