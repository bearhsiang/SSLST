#!/usr/local/bin bash

dataset=libritrans-en-fr
dataset_root=/hdd/libritrans
data_root=data
tsv_root=data/tsv
key=audio

hubert_root=/home/sean/Desktop/SSLST/fairseq/examples/hubert
ckpt_path=data/ssl_models/hubert_base_ls960.pt
km_path=data/kmeans_model/hubert_base_ls960_L9_km500.bin
ckpt_name=hubert
layer=9
km=500
nshard=5

name=${ckpt_name}_l${layer}_km${km}
feat_dir=/hdd/ssl_feat/$ckpt_name-$layer

### create manifest
for split in test dev train; do
    python utils_new/create_fs_manifest.py \
        --audio-root $dataset_root/$split/audiofiles \
        --tsv-file $tsv_root/$dataset/$split.tsv \
        --key $key \
        --output $data_root/$dataset/manifest/$split.tsv
done

### apply km quantization
for split in train dev test; do
    for rank in `seq 0 $((nshard-1))`; do
        python $hubert_root/simple_kmeans/dump_hubert_feature.py \
            $data_root/$dataset/manifest \
            $split \
            $ckpt_path \
            $layer \
            $nshard \
            $rank \
            $feat_dir
        python $hubert_root/simple_kmeans/dump_km_label.py \
            $feat_dir \
            $split \
            $km_path \
            $nshard \
            $rank \
            $data_root/$dataset/$name-tmp
    done
    for rank in `seq 0 $((nshard-1))`; do
        cat $data_root/$dataset/$name-tmp/${split}_${rank}_${nshard}.km \
            >> $data_root/$dataset/$split.$name
    done
done