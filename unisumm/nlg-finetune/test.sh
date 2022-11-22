export MODEL_PATH=../few-shot-unisumm/$1/$2/$7-$3/ckpt-$8
export SAVE_PATH=../unisumm_outs/$2.$7-$3
export TASK_MAP_FILE=../few-shot-unisumm/$1/$2/$7-$3/cached_features_for_training/task_map.json
export INPUT_FILE=../data/Summzoo/$2/test/$2.test.bart.uncased.jsonl

CUDA_VISIBLE_DEVICES=$6 python decode_seq2seq.py   --fp16  --do_lower_case  --model_path $MODEL_PATH --max_seq_length 2048 --max_tgt_length $4 --batch_size 4 --beam_size 5   --length_penalty 0.6 --mode s2s  --min_len $5 --input_file $INPUT_FILE  --decode_task $1 --task_map_file $TASK_MAP_FILE   --output_file  $SAVE_PATH
