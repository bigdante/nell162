LR=1e-4
PRE_SEQ_LEN=64
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
HOST_FILE_PATH="hostfile"
export TORCH_EXTENSIONS_DIR=/zhangpai22/xll/torch_extension
deepspeed --master_port $MASTER_PORT --hostfile ${HOST_FILE_PATH} main.py \
  --deepspeed deepspeed.json \
  --do_train \
  --train_file AdvertiseGen/train.json \
  --test_file AdvertiseGen/dev.json \
  --prompt_column content \
  --response_column summary \
  --overwrite_cache \
  --model_name_or_path THUDM/chatglm-6b \
  --output_dir ./output/adgen-chatglm-6b-test-$LR \
  --overwrite_output_dir \
  --max_source_length 64 \
  --max_target_length 64 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --predict_with_generate \
  --max_steps 5000 \
  --logging_steps 10 \
  --save_steps 1000 \
  --learning_rate $LR \
  --fp16 \
  --pre_seq_len $PRE_SEQ_LEN
