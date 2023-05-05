#!/bin/bash
source ./script/model_t5_base.sh
#source ./script/model_t5_small.sh
echo "===============$model_name max_length:$max_length==============="
relation_list=(
  "country of citizenship"
  "date of birth"
  "place of birth"
  "participant of"
  "located in the administrative territorial entity"
  "contains administrative territorial entity"
  "participant"
  "location"
  "followed by"
  "country"
  "educated at"
  "date of death"
  "sibling"
  "head of government"
  "legislative body"
  "conflict"
  "applies to jurisdiction"
  "instance of"
  "performer"
  "publication date"
  "creator"
  "author"
  "composer"
  "lyrics by"
  "member of"
  "notable work"
  "inception"
  "part of"
  "cast member"
  "director"
  "has part"
  "production company"
  "owned by"
  "headquarters location"
  "developer"
  "manufacturer"
  "country of origin"
  "publisher"
  "parent organization"
  "subsidiary"
  "capital of"
  "capital"
  "spouse"
  "father"
  "child"
  "religion"
  "mother"
  "located in or next to body of water"
  "located on terrain feature"
  "basin country"
  "member of political party"
  "mouth of the watercourse"
  "place of death"
  "military branch"
  "work location"
  "start time"
  "award received"
  "point in time"
  "founded by"
  "employer"
  "head of state"
  "member of sports team"
  "league"
  "present in work"
  "position held"
  "chairperson"
  "languages spoken, written or signed"
  "location of formation"
  "operator"
  "producer"
  "record label"
  "follows"
  "replaced by"
  "replaces"
  "end time"
  "subclass of"
  "residence"
  "sister city"
  "original network"
  "ethnic group"
  "separated from"
  "screenwriter"
  "continent"
  "platform"
  "product or material produced"
  "genre"
  "series"
  "narrative location"
  "parent taxon"
  "original language of work"
  "dissolved, abolished or demolished"
  "territory claimed by"
  "characters"
  "influenced by"
  "official language"
  "unemployment rate"
)
function is_port_available() {
  local port=$1
  if [ "$(lsof -i :$port | wc -l)" == "0" ]; then
    return 0
  else
    return 1
  fi
}

relation_list=(
  "country of citizenship"
)
model_name=$model_name
export OMP_NUM_THREADS=$(($(nproc) / 2))
save_ckpt_path=./ckpt
for relation in "${relation_list[@]}"; do
  mkdir -p ${save_ckpt_path}/"$relation"
  while true; do
    RANDOM_PORT=$((10000 + RANDOM % 50000))
    if is_port_available $RANDOM_PORT; then
      break
    else
      echo "Port $RANDOM_PORT is in use, trying another one"
    fi
  done
  echo "===============Running vf finetune experiment for relation: $relation==============="
  CUDA_VISIBLE_DEVICES=7 torchrun \
    --master_port=$RANDOM_PORT \
    --nnodes=1 \
    --nproc_per_node=1 \
    bert_train.py \
    --train \
    --max_length=$max_length \
    --model_name=${model_name} \
    --epochs=10 \
    --batch_size=8 \
    --train_data_path=./for_vf_train_data/"$relation".json \
    --save_ckpt_path=${save_ckpt_path}/"$relation" \
    --experiment="$relation"
done
