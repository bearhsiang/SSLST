#!/usr/local/bin bash

hidden=hubert-50-simple-6

input_dir=data/u2t/libritrans-en-fr/hubert-km50-l6-simple
src_key=hidden_unit
tgt_key=tgt_text
src_lang=hubert_km50_l6_simple
tgt_lang=fr

output_dir=data/trans/libritrans-en-fr/$src_lang-$tgt_lang

tgt_bpe=data/s2t/libritrans-en-fr/normalized/bpe-tgt-bpe-8000.model

mkdir -p $output_dir

for split in test dev train; do
    python convert/trans.py \
        --input-tsv $input_dir/$split.tsv \
        --output-prefix $output_dir/$split \
        --src-key $src_key \
        --src-lang $src_lang \
        --tgt-key $tgt_key \
        --tgt-lang $tgt_lang \
        --tgt-bpe $tgt_bpe
done 

cp $tgt_bpe $output_dir