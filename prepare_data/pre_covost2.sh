#!/usr/local/bin bash

source script/setup.sh

src_lang=de
tgt_lang=en

filename=covost_v2.${src_lang}_${tgt_lang}.tsv

if [ ! -f $sslst_covost_tsv_root/$filename.tar.gz ]; then
    wget -P $sslst_covost2_tsv_root https://dl.fbaipublicfiles.com/covost/$filename.tar.gz
    tar -zxvf $sslst_covost2_tsv_root/$filename.tar.gz --directory $sslst_covost2_tsv_root
fi

python $sslst_covost_root/get_covost_splits.py \
  --version 2 --src-lang $src_lang --tgt-lang $tgt_lang \
  --root $sslst_covost2_tsv_root \
  --cv-tsv $sslst_covost2_root/$src_lang/validated.tsv