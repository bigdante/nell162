PRE_SEQ_LEN=64

CUDA_VISIBLE_DEVICES=7 python3 web_demo.py \
    --model_name_or_path THUDM/chatglm-6b \
    --ptuning_checkpoint output/adgen-chatglm-6b-pt-64-1e-2/checkpoint-6 \
    --pre_seq_len $PRE_SEQ_LEN

