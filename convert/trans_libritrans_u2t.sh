#!/usr/local/bin bash

input_dir=data/u2t/libritrans-en-fr/libritrans_en_fr-train-0.01-hubert-6-50-simple
src_key=hidden_unit
tgt_key=tgt_text
src_lang=lt_0.1_hubert_6_50_simple
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