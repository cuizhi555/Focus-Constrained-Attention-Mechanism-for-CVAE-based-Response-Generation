#!/bin/bash

base_dir=$(cd "$(dirname "$0")";pwd)
parentdir="$(dirname "$base_dir")"

#GPUS should be assigned as "0,1"
GPUS=$1


export CUDA_VISIBLE_DEVICES=$GPUS


DIRECTION=src2tgt
DATA_DIR=$base_dir/gen_data

PARAM_SET=base
MODEL_DIR=$base_dir/model_${PARAM_SET}

steps_between_evals=5000
train_steps=200000
#train_epochs=10
#epochs_between_evals=1

mkdir -p $MODEL_DIR

vocab_size=50000
from_lan=src
to_lan=tgt

cd $parentdir
cd focusseq 


python -m official.seq2seq.seq2seq_main \
    --data_dir=$DATA_DIR \
    --model_dir=$MODEL_DIR \
    --param_set=$PARAM_SET \
    --fro=$from_lan \
    --to=$to_lan \
    --train_steps=${train_steps} \
    --steps_between_evals=${steps_between_evals} \
    --vocabulary=$vocab_size \
    --save_checkpoints_secs 240 \
    --gpus=${GPUS} \
    --keep_checkpoint_max=20 \


