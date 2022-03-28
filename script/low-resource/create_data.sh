#!/usr/local/bin bash

source script/setup.sh

dataset=$1
src_lang=$2
tgt_lang=$3
hr=$4

ori_dir=$sslst_data_root/$dataset
tgt_dir=$sslst_data_root/$dataset-hr$hr

bpe=8000

mkdir -p $tgt_dir

# selected ids
python utils_new/create_selected_ids.py \
    --manifest-file $ori_dir/manifest/train.tsv \
    --hour $hr \
    --output $tgt_dir/selected_ids.txt \
    --seed $sslst_seed

# manifest
python utils_new/select_with_ids.py \
    --selected-ids-file $tgt_dir/selected_ids.txt \
    --input $ori_dir/manifest/train.tsv \
    --output $tgt_dir/manifest/train.tsv \
    --has-header

for split in test dev; do
    cp $ori_dir/manifest/$split.tsv $tgt_dir/manifest/$split.tsv
done

# text
for lang in $src_lang $tgt_lang; do
    python utils_new/select_with_ids.py \
        --selected-ids-file $tgt_dir/selected_ids.txt \
        --input $ori_dir/tmp/train.filtered.$lang \
        --output $tgt_dir/tmp/train.filtered.$lang

    for split in dev test; do
        cp $ori_dir/tmp/$split.filtered.$lang $tgt_dir/tmp/$split.filtered.$lang
    done

done

for lang in $src_lang $tgt_lang; do

    spm_train \
        --input=$tgt_dir/tmp/train.filtered.$lang \
        --model_prefix=$tgt_dir/tmp/$lang-$bpe \
        --vocab_size=$bpe \
        --character_coverage=1

done

for lang in $src_lang $tgt_lang; do

    for split in dev test train; do

        spm_encode \
            --model=$tgt_dir/tmp/$lang-$bpe.model \
            --output_format=piece \
            < $tgt_dir/tmp/$split.filtered.$lang \
            > $tgt_dir/$split.$lang

    done
    
done