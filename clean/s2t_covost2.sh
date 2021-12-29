#!/usr/local/bin bash

input_dir=data/s2t/covost2-en-de/raw
output_dir=data/s2t/covost2-en-de/normalized

src_lang=en
tgt_lang=de

for split in train dev; do
    # source text
    python clean/normalize_tsv.py \
        --normalize \
        --lowercase \
        --remove-punctuation \
        --remove-consecutive-blank \
        -i $input_dir/$split.tsv \
        -o $output_dir/.$split.tsv \
        -k src_text \
        -L $src_lang

    # target text
    python clean/normalize_tsv.py \
        --normalize \
        --remove-consecutive-blank \
        -i $output_dir/.$split.tsv \
        -o $output_dir/$split.tsv \
        -k tgt_text \
        -L $tgt_lang

    rm -rf $output_dir/.$split.tsv
done

for split in test; do
    cp $input_dir/$split.tsv $output_dir/$split.tsv
done