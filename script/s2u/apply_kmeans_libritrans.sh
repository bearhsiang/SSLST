#!/usr/local/bin bash 

source script/setup.sh

dataset=libritrans-en-de
splits="train dev test"
audio_key=audio

model=$1
layer=$2
n_cluster=$3
km_tag=ls

km_model_path=$sslst_data_root/kmeans_model/$model-$km_tag-L$layer-km$n_cluster.bin
suffix=${model}_l${layer}_${km_tag}${n_cluster}

for split in $splits; do
    bash script/s2u/apply_kmeans.sh \
        --dataset $dataset \
        --split $split \
        --audio-key $audio_key \
        --audio-root $sslst_libritrans_root \
        --model $model \
        --layer $layer \
        --km-model-path $km_model_path \
        --suffix $suffix
done