#!/usr/local/bin bash

source script/setup.sh

dataset=libritrans-en-fr
src_key=src_text
tgt_key=tgt_text
src_lang=en
tgt_lang=fr

tmp_dir=$sslst_data_root/$dataset/tmp

cleaner=$sslst_mosesdecoder_root/scripts/tokenizer
filter=$sslst_mosesdecoder_root/scripts/training

bpe=8000

for split in test dev train; do
    
    python utils_new/extract_tsv_field.py \
        --input $sslst_data_root/tsv/$dataset/$split.tsv \
        --key $src_key \
        --output $tmp_dir/$split.raw.$src_lang

    python utils_new/extract_tsv_field.py \
        --input $sslst_data_root/tsv/$dataset/$split.tsv \
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
        --input=$tmp_dir/train.filtered.$lang \
        --model_prefix=$tmp_dir/$lang-$bpe \
        --vocab_size=$bpe \
        --character_coverage=1

done

for lang in $src_lang $tgt_lang; do

    for split in dev test train; do

        spm_encode \
            --model=$tmp_dir/$lang-$bpe.model \
            --output_format=piece \
            < $tmp_dir/$split.filtered.$lang \
            > $sslst_data_root/$dataset/$split.$lang

    done
    
done

