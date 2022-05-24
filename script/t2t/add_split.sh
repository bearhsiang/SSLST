#!/usr/local/bin bash

mose_root=mosesdecoder
data_root=data
data_bin_root=data-bin

dataset=libritrans-en-fr
split=newscrawl_1M
split_data_prefix=$data_root/$dataset/newscrawl-fr-1M/back
src_lang=hubert_l9_km500_simple
tgt_lang=fr

cleaner=$mose_root/scripts/tokenizer
bpe=8000

data_dir=$data_root/$dataset
origin_data_bin_dir=$data_bin_root/$dataset/$src_lang-$tgt_lang
dest_data_bin_dir=$data_bin_root/$dataset-$split/$src_lang-$tgt_lang

cp $split_data_prefix.$src_lang $data_dir/$split.$src_lang

# normalize punctuation and remove unprintable char
perl $cleaner/normalize-punctuation.perl < $split_data_prefix.$tgt_lang \
| perl $cleaner/remove-non-printing-char.perl \
| perl $cleaner/lowercase.perl \
| spm_encode --model $data_dir/tmp/$tgt_lang-$bpe.model --output_format=piece \
> $data_dir/$split.$tgt_lang

# combine with original set
for lang in $src_lang $tgt_lang; do
    cat $data_dir/train.$src_lang $data_dir/$split.$lang > $data_dir/train-$split.$lang 
done

fairseq-preprocess \
    -s $src_lang \
    -t $tgt_lang \
    --trainpref $data_dir/train-$split \
    --validpref $data_dir/dev \
    --testpref $data_dir/test \
    --destdir $dest_data_bin_dir \
    --srcdict $origin_data_bin_dir/dict.$src_lang.txt \
    --tgtdict $origin_data_bin_dir/dict.$tgt_lang.txt 

