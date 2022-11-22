export TRAIN_FILE=../data/Summzoo/$2/$4-shot/train.jsonl.h$4.s$3
export OUTPUT_DIR=../few-shot-unisumm/$1/$2/$4-$3
export CACHE_DIR=../cache/unisumm
export LOAD_FROM=../unisumm_model/ckpt-300000
export TASK_MAP_FILE=../unisumm_model/task_map.json

rm $OUTPUT_DIR -rf

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch  --master_port 8888  --nproc_per_node 4  \run_seq2seq.py   --train_file $TRAIN_FILE --output_dir $OUTPUT_DIR  --model_type bart --model_name_or_path facebook/bart-large --fp16 --fp16_opt_level O2   --max_source_seq_length 2048 --max_target_seq_length 400 --per_gpu_train_batch_size 2 --gradient_accumulation_steps 4   --learning_rate 1.5e-4 --num_warmup_steps $5 --num_training_steps $6  --cache_dir $CACHE_DIR --save_steps $6 --num_shards 1   --num_prefixs_per_task 256    --tasks  $1 --logging_steps 50 --use_prefix 1 --only_train_prefix 1   --load_from $LOAD_FROM   --task_map_file $TASK_MAP_FILE
