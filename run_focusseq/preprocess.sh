#!/bin/bash

base_dir=$(cd "$(dirname "$0")";pwd)
parentdir="$(dirname "$base_dir")"

DIRECTION=src2tgt
PROBLEM=dialog_${DIRECTION}_30k
DATA_DIR=$base_dir/gen_data/
RAW_DIR=$parentdir/data

echo "--------------------"
echo "problem: ${PROBLEM}"
echo "data dir: ${DATA_DIR}"
echo "raw dir: ${RAW_DIR}"
echo "--------------------"

cd ..

vocab_size=50000
from_lan=src
to_lan=tgt

mkdir -p $DATA_DIR

cd $parentdir
cd focusseq 


python -m official.seq2seq.data_download \
  --rd=$RAW_DIR \
  --dd=$DATA_DIR \
  --search=$vocab_size \
  --pp=$PROBLEM \
  --fro=$from_lan \
  --to=$to_lan

