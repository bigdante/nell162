#!/bin/bash
PRE_SEQ_LEN=64
LR=1e-2
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
HOST_FILE_PATH="hostfile"
export TORCH_EXTENSIONS_DIR=/zhangpai22/xll/torch_extension
gpt_options=" \
  --deepspeed deepspeed.json \
  --do_train \
  --train_file auto_kg/one/train.json \
  --validation_file auto_kg/one/dev.json \
  --prompt_column prompt \
  --response_column response \
  --history_column history \
  --overwrite_cache \
  --model_name_or_path THUDM/chatglm-6b \
  --output_dir ./output/adgen-chatglm-6b-pt-one_new-$PRE_SEQ_LEN-$LR \
  --overwrite_output_dir \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --predict_with_generate \
  --max_steps 120000 \
  --logging_steps 50 \
  --save_steps 1000 \
  --learning_rate $LR \
  --pre_seq_len $PRE_SEQ_LEN \
  --fp16
"

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
run_cmd="TORCH_HOME=/zhangpai22/xll/torch_cache TRANSFORMERS_CACHE=/zhangpai22/xll/.cache/ ${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port ${MASTER_PORT} --hostfile ${HOST_FILE_PATH} main.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
