#!/usr/local/bin bash 

source script/setup.sh

dataset=covost2-de-en
splits="train dev test"
audio_key=path
model=$1
layer=$2
n_cluster=$3
src_lang=de
tgt_lang=en
km_model_path=$sslst_data_root/kmeans_model/$model-librispeech-train-clean-100-L$layer-km$n_cluster.bin
suffix=${model}_l${layer}_km${n_cluster}

for split in $splits; do
    bash script/s2u/apply_kmeans.sh \
        --dataset $dataset \
        --split $split \
        --audio-key $audio_key \
        --audio-root $sslst_covost2_data_root/$sslst_cv_version/$lang/clips \
        --model $model \
        --layer $layer \
        --km-model-path $km_model_path \
        --suffix $suffix
done