#!/usr/local/bin bash

source script/setup.sh

dataset=libritrans-en-fr
arch=transformer_iwslt_de_en
bt_name=newscrawl_fr
bt_subset=max30-min10-N10000
src_lang=hubert_l9_km500_simple
tgt_lang=fr

dest_dir=$sslst_data_root/$dataset/$bt_name

# fairseq-generate \
#     $sslst_data_bin_root/$dataset/$bt_name-$bt_subset/$src_lang-$tgt_lang \
#     -s $tgt_lang -t $src_lang \
#     --gen-subset test \
#     --path $sslst_output_root/$dataset/$arch-$tgt_lang-$src_lang/checkpoint_best.pt \
#     --beam 1 \
#     --max-len-a 10 \
#     --max-len-b 10 | tee $dest_dir/$bt_subset.gen

cat $dest_dir/$bt_subset.gen | grep ^S | LC_ALL=C sort -V | cut -f2- > $dest_dir/$bt_subset.gen.$tgt_lang
cat $dest_dir/$bt_subset.gen | grep ^D | LC_ALL=C sort -V | cut -f3- > $dest_dir/$bt_subset.gen.$src_lang