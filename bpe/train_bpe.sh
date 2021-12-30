#!/usr/local/bin bash

input_dir=data/s2t/libritrans-en-fr/normalized
pattern="train*"
model_type=char
keys=tgt_text

output_prefix=$input_dir/bpe-tgt-$model_type

python bpe/train_bpe.py \
    -i $input_dir \
    -p "$pattern" \
    --keys $keys \
    -o $output_prefix \
    --model-type $model_type 