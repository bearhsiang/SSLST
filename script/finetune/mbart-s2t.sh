#!/usr/local/bin bash

source script/setup.sh

mbart_dir=$sslst_data_root/mbart.cc25.v2

name=libritrans-en-fr
splits="train dev test"
model_name=hubert
model_dim=768
layer=9
src_lang=en
src_lang_tag=en_XX
tgt_lang=fr
tgt_lang_tag=fr_XX

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

dest_dir=$sslst_data_root/$name/s2t-mbart-${model_name}_l$layer-$tgt_lang
mkdir $dest_dir

for split in $splits; do
    python utils_new/create_s2t_data.py \
        --prefix $sslst_data_root/$name/tmp/$split.filtered \
        --src-lang $src_lang \
        --src-lang-tag $src_lang_tag \
        --tgt-lang $tgt_lang \
        --tgt-lang-tag $tgt_lang_tag \
        --split $split \
        --manifest $sslst_data_root/$name/manifest/$split.tsv \
        --feat-root $sslst_feat_root/$name/$model_name/$layer \
        --output $dest_dir/$split.tsv
done

cp $mbart_dir/dict.txt $dest_dir/dict.txt
cp $mbart_dir/sentence.bpe.model $dest_dir/sentence.bpe.model

python utils_new/create_s2t_config.py \
    --vocab-filename dict.txt \
    --bpe-model $dest_dir/sentence.bpe.model \
    --audio-root $sslst_feat_root/$name/$model_name/$layer \
    --input-feat-per-channel $model_dim \
    --output $dest_dir/config.yaml
