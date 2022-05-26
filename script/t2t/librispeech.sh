#!/usr/local/bin bash

source script/setup.sh

dataset=librispeech
src_key=transcript
src_lang=en
splits="train-clean-100 dev-clean test-clean"

tmp_dir=$sslst_data_root/$dataset/tmp

cleaner=$sslst_mosesdecoder_root/scripts/tokenizer
filter=$sslst_mosesdecoder_root/scripts/training

bpe=8000

for split in $splits; do
    
    python utils_new/extract_tsv_field.py \
        --input $sslst_data_root/tsv/$dataset/$split.tsv \
        --key $src_key \
        --output $tmp_dir/$split.raw.$src_lang

done


for split in $splits; do

    for lang in $src_lang; do

        # normalize punctuation, remove unprintable char and lowercase
        perl $cleaner/normalize-punctuation.perl < $tmp_dir/$split.raw.$lang \
        | perl $cleaner/remove-non-printing-char.perl \
        | perl $cleaner/lowercase.perl \
        > $tmp_dir/$split.clean.$lang
    
    done

    cp $tmp_dir/$split.clean.$src_lang $tmp_dir/$split.filtered.$src_lang

done

for lang in $src_lang; do

    spm_train \
        --input=$tmp_dir/train-clean-100.filtered.$lang \
        --model_prefix=$tmp_dir/$lang-$bpe \
        --vocab_size=$bpe \
        --character_coverage=1

done

for lang in $src_lang; do

    for split in $splits; do

        spm_encode \
            --model=$tmp_dir/$lang-$bpe.model \
            --output_format=piece \
            < $tmp_dir/$split.filtered.$lang \
            > $sslst_data_root/$dataset/$split.$lang

    done
    
done

