#!/usr/local/bin bash

input_dir=data/s2t/libritrans-en-fr/normalized
pattern="train*"
model_type=bpe
vocab_size=8000
coverage=1.0

for lang in src tgt; do
    output_prefix=$input_dir/bpe-$lang-$model_type-$vocab_size
    python bpe/train_bpe.py \
        -i $input_dir \
        -p "$pattern" \
        --keys ${lang}_text \
        -o $output_prefix \
        --model-type $model_type \
        --vocab-size $vocab_size \
        --character-coverage $coverage
done