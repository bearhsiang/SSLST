#!/usr/local/bin bash

mose_root=mosesdecoder

dataset=libritrans-en-fr
data_root=data
src_key=src_text
tgt_key=tgt_text
src_lang=en
tgt_lang=fr

tmp_dir=$data_root/$dataset/tmp
data_dir=$data_root/$dataset
tsv_dir=$data_root/tsv/$dataset

cleaner=$mose_root/scripts/tokenizer
filter=$mose_root/scripts/training

bpe=8000

for split in test dev train; do
    
    python utils_new/extract_tsv_field.py \
        --input $tsv_dir/$split.tsv \
        --key $src_key \
        --output $tmp_dir/$split.raw.$src_lang

    python utils_new/extract_tsv_field.py \
        --input $tsv_dir/$split.tsv \
        --key $tgt_key \
        --output $tmp_dir/$split.raw.$tgt_lang

done


for split in test dev train; do

    for lang in $src_lang $tgt_lang; do

        # normalize punctuation and remove unprintable char
        perl $cleaner/normalize-punctuation.perl < $tmp_dir/$split.raw.$lang | perl $cleaner/remove-non-printing-char.perl > $tmp_dir/$split.clean.$lang
    
    done
    
    # lowercase
    perl $filter/clean-corpus-n.perl \
        $tmp_dir/$split.clean \
        $src_lang $tgt_lang \
        $tmp_dir/$split.filtered \
        1 1000 \
        --ignore-ratio \
        --lc

done

for lang in $src_lang $tgt_lang; do

    spm_train \
        --input $tmp_dir/train.filtered.$lang \
        --model_prefix $tmp_dir/$lang-$bpe \
        --vocab_size $bpe \
        --character_coverage 1

done

for lang in $src_lang $tgt_lang; do

    for split in dev test train; do

        spm_encode \
            --model $tmp_dir/$lang-$bpe.model \
            --output_format=piece \
            < $tmp_dir/$split.filtered.$lang \
            > $data_dir/$split.$lang

    done
    
done

