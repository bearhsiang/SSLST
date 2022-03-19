#!/usr/local/bin/bash

source script/setup.sh

dataset=libritrans-en-fr
splits="train dev test"
src_lang=en
tgt_lang=fr
audio_root=$sslst_libritrans_root

bpe=8000

orig_dir=$sslst_data_root/$dataset
dest_dir=$sslst_data_root/$dataset/s3prl

mkdir -p $dest_dir

for split in $splits; do
    python utils_new/create_s2t_data.py \
        --prefix $orig_dir/tmp/$split.filtered \
        --src-lang $src_lang \
        --tgt-lang $tgt_lang \
        --manifest $orig_dir/manifest/$split.tsv \
        --split $split \
        --output $dest_dir/$split.tsv
done

for lang in $src_lang $tgt_lang; do
    python utils_new/create_s2t_dict.py \
        --in-dict $orig_dir/tmp/$lang-$bpe.vocab \
        --out-dict $dest_dir/$lang-$bpe.txt
    cp $orig_dir/tmp/$lang-$bpe.model $dest_dir/$lang-$bpe.model
done

python utils_new/create_s2t_config.py \
    --vocab-filename $dest_dir/$tgt_lang-$bpe.txt \
    --bpe-model $dest_dir/$tgt_lang-$bpe.model \
    --audio-root $audio_root \
    --use-audio-input \
    --shuffle \
    --output $dest_dir/config.yaml