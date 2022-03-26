#!/usr/local/bin bash

source script/setup.sh

name=$1
model_name=$2
model_dim=$3 # hubert/wav2vec2: 768
layer=$4
src_lang=$5
tgt_lang=$6

splits="train dev test"

bpe_model_prefix=$tgt_lang-8000

tgt_dir=$sslst_data_root/$name/s2t-${model_name}_l$layer-$tgt_lang
mkdir $tgt_dir

for split in $splits; do
    python utils_new/create_s2t_data.py \
        --prefix $sslst_data_root/$name/tmp/$split.filtered \
        --src-lang $src_lang \
        --tgt-lang $tgt_lang \
        --split $split \
        --manifest $sslst_data_root/$name/manifest/$split.tsv \
        --feat-root $sslst_feat_root/$name/$model_name/$layer \
        --output $tgt_dir/$split.tsv
done

python utils_new/create_s2t_dict.py \
    --in-dict $sslst_data_root/$name/tmp/$bpe_model_prefix.vocab \
    --out-dict $tgt_dir/dict.txt

cp $sslst_data_root/$name/tmp/$bpe_model_prefix.model $tgt_dir/$bpe_model_prefix.model

python utils_new/create_s2t_config.py \
    --vocab-filename dict.txt \
    --bpe-model $bpe_model_prefix.model \
    --audio-root $sslst_feat_root/$name/$model_name/$layer \
    --input-feat-per-channel $model_dim \
    --output $tgt_dir/config.yaml
