PRE_SEQ_LEN=12
CHECKPOINT=adgen-chatglm-6b-pt-12-2e-2
STEP=3

CUDA_VISIBLE_DEVICES=7 python3 main.py \
    --do_predict \
    --validation_file auto_kg/dev.json \
    --test_file auto_kg/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path THUDM/chatglm-6b \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
