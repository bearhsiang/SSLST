#!/usr/local/bin bash

source script/setup.sh

dataset=libritrans-en-fr
src_lang=hubert_l9_km500_simple
tgt_lang=fr
bpe=8000

bt_name=newscrawl_fr
bt_subset=max30-min10-N10000
file=$sslst_data_root/$bt_name/$bt_subset.txt

dest_dir=$sslst_data_root/$dataset/$bt_name
mkdir $dest_dir -p

cleaner=$sslst_mosesdecoder_root/scripts/tokenizer

perl $cleaner/normalize-punctuation.perl < $file \
    | perl $cleaner/remove-non-printing-char.perl \
    | perl $cleaner/lowercase.perl \
    | spm_encode --model $sslst_data_root/$dataset/tmp/$tgt_lang-$bpe.model --output_format piece \
    > $dest_dir/$bt_subset.$tgt_lang

cp $dest_dir/$bt_subset.$tgt_lang $dest_dir/$bt_subset.$src_lang

fairseq-preprocess \
    -s $src_lang \
    -t $tgt_lang \
    --testpref $dest_dir/$bt_subset \
    --destdir $sslst_data_bin_root/$dataset/$bt_name-$bt_subset/$src_lang-$tgt_lang \
    --srcdict $sslst_data_bin_root/$dataset/$src_lang-$tgt_lang/dict.$src_lang.txt \
    --tgtdict $sslst_data_bin_root/$dataset/$src_lang-$tgt_lang/dict.$tgt_lang.txt 