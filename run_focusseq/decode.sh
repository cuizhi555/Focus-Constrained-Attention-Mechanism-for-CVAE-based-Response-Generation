#!/bin/bash

base_dir=$(cd "$(dirname "$0")";pwd)
parentdir="$(dirname "$base_dir")"

export CUDA_VISIBLE_DEVICES=$1

DIRECTION=srctgt
BASE_DIR=$base_dir
PARAM_SET=base
DATA_DIR=${BASE_DIR}/gen_data
MODEL_DIR=$base_dir/model_base
#MODEL_DIR=$base_dir/${VERSION}/model_${PARAM_SET}

IN_PATH=$parentdir/data/test.50000.src
DEC_PATH=$parentdir/data/test.50000.gen

vocab_size=50000


cd $parentdir
cd focusseq

python -m official.seq2seq.translate \
    -dd $DATA_DIR \
    --file ${IN_PATH} \
    --file_out ${DEC_PATH} \
    -md ${MODEL_DIR} \
    -mp $PARAM_SET \
    --search $vocab_size \
    --fro src \
    --to  tgt \
    --beam_width 30

