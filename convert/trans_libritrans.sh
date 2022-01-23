#!/usr/local/bin bash

input_dir=data/s2t/libritrans-en-fr/normalized
src_key=src_text
tgt_key=tgt_text
src_lang=en
tgt_lang=fr

output_dir=data/trans/libritrans-en-fr/$src_lang-$tgt_lang

src_bpe=data/s2t/libritrans-en-fr/normalized/bpe-src-bpe-8000.model
tgt_bpe=data/s2t/libritrans-en-fr/normalized/bpe-tgt-bpe-8000.model

mkdir -p $output_dir

for split in test dev train; do
    python convert/trans.py \
        --input-tsv $input_dir/$split.tsv \
        --output-prefix $output_dir/$split \
        --src-key $src_key \
        --src-lang $src_lang \
        --src-bpe $src_bpe \
        --tgt-key $tgt_key \
        --tgt-lang $tgt_lang \
        --tgt-bpe $tgt_bpe
done 

cp $src_bpe $output_dir
cp $tgt_bpe $output_dir