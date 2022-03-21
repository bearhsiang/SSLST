#!/usr/local/bin bash

source script/setup.sh

src_lang=de
tgt_lang=en
dataset=covost2-$src_lang-$tgt_lang
src_key=sentence
tgt_key=translation

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

        # normalize punctuation, remove unprintable char and lowercase
        perl $cleaner/normalize-punctuation.perl < $tmp_dir/$split.raw.$lang \
        | perl $cleaner/remove-non-printing-char.perl \
        | perl $cleaner/lowercase.perl \
        > $tmp_dir/$split.clean.$lang
    
    done

    cp $tmp_dir/$split.clean.$src_lang $tmp_dir/$split.filtered.$src_lang
    cp $tmp_dir/$split.clean.$tgt_lang $tmp_dir/$split.filtered.$tgt_lang

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

