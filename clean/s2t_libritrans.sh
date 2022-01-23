#!/usr/local/bin bash

input_dir=data/s2t/libritrans-en-fr/raw
output_dir=data/s2t/libritrans-en-fr/normalized

src_lang=en
tgt_lang=fr

for split in train dev test; do
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

# for split in test; do
#     cp $input_dir/$split.tsv $output_dir/$split.tsv
# done