#!/usr/local/bin bash

# config
data_root=data
name=libritrans-en-fr
input_dir=$data_root/s2t/$name/normalized
output_dir=$data_root/s2u/$name
audio_root=/hdd/libritrans
audio_key=audio

### config for quantization
feature_type=fbank
layer=0
n_clusters=50
km_models_dir=data/kmeans_model
km_model=libritrans_en_fr-train-0.01-$feature_type-$layer-$n_clusters
batch_size=1

### config for reduction
mode=simple

for split in train test dev; do

    python convert/s2u_quantize.py \
        -u $feature_type \
        -i $input_dir/$split.tsv \
        -k $audio_key \
        -d $audio_root/$split/audiofiles \
        -l $layer \
        --kmeans-model $km_models_dir/$km_model \
        --output $output_dir/$km_model/$split.txt \
        --cuda \
        --batch_size $batch_size

    python convert/s2u_reduce.py \
        --input $output_dir/$km_model/$split.txt \
        --output $output_dir/$km_model-$mode/$split.txt \
        --mode $mode

done


