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
n_cluster=500
nshard=5

feat_dir=/hdd/ssl_feat/$dataset/$ckpt_name/$layer

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
for rank in `seq 0 $((nshard-1))`; do
    if [ -f $feat_dir/${split}_${rank}_${nshard}.npy ] && [ -f $feat_dir/${split}_${rank}_${nshard}.len ]; then
        echo $feat_dir/${split}_${rank}_${shard}.[npy,len] exist, skip
    else
        python $hubert_root/simple_kmeans/dump_hubert_feature.py \
            $data_root/$dataset/manifest \
            $split \
            $ckpt_path \
            $layer \
            $nshard \
            $rank \
            $feat_dir
    fi
done

km_path=$data_root/kmeans_model/$ckpt_name-$dataset-$split-L$layer-km$n_cluster.bin

python $hubert_root/simple_kmeans/learn_kmeans.py \
    $feat_dir \
    $split \
    $nshard \
    $km_path \
    $n_cluster
    --percent 0.1