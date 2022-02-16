#!/usr/local/bin bash

dataset=librispeech
split=train-clean-100
dataset_root=/hdd/LibriSpeech
data_root=data
tsv_root=data/tsv
key=audio

hubert_root=/home/sean/Desktop/SSLST/fairseq/examples/hubert
ckpt_path=data/ssl_models/hubert_base_ls960.pt
ckpt_name=hubert
layer=6
km=500
nshard=5

name=${ckpt_name}_l${layer}_km${km}
feat_dir=/hdd/ssl_feat/$ckpt_name-$layer

### create manifest
echo "create manifest"
if [ ! -f "$data_root/$dataset/manifest/$split.tsv" ]; then
    python utils_new/create_fs_manifest.py \
        --audio-root $dataset_root \
        --tsv-file $tsv_root/$dataset/$split.tsv \
        --key $key \
        --output $data_root/$dataset/manifest/$split.tsv
else
    echo "$data_root/$dataset/manifest/$split.tsv exists, skip"
fi

### dump features
# for rank in `seq 0 $((nshard-1))`; do
#     python $hubert_root/simple_kmeans/dump_hubert_feature.py \
#         $data_root/$dataset/manifest \
#         $split \
#         $ckpt_path \
#         $layer \
#         $nshard \
#         $rank \
#         $feat_dir
# done