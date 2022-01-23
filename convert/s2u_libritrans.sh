#!/usr/local/bin bash

# config
data_root=data
name=libritrans-en-fr
input_dir=$data_root/s2t/$name/normalized
output_dir=$data_root/s2u/$name
audio_root=/hdd/libritrans
audio_key=audio

### config for quantization
feature_type=hubert
n_cluster=50
km_models_dir=data/km_models
layer=6
ext=.wav
batch_size=1

### config for reduction
mode=simple

### 

for split in train test dev; do

    # python convert/s2u_quantize.py \
    #     -u $feature_type \
    #     -i $input_dir/$split.tsv \
    #     -k $audio_key \
    #     -d $audio_root/$split/audiofiles \
    #     -l $layer \
    #     --kmeans-model $km_models_dir/$feature_type-$n_cluster.bin \
    #     --output $output_dir/$feature_type-$n_cluster-$layer/$split.txt \
    #     --cuda \
    #     --batch_size $batch_size


    # python $fairseq_root/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    #     --feature_type $feature_type \
    #     --kmeans_model_path $km_models_dir/$feature_type-$n_cluster.bin \
    #     --acoustic_model_path $ac_models_dir/$feature_type.pt \
    #     --layer $layer \
    #     --manifest_path $output_dir/manifest/$split.txt \
    #     --out_quantized_file_path $output_dir/quantized-$feature_type-$n_cluster-$layer/$split.txt \
    #     --extension $ext

    python convert/s2u_reduce.py \
        --input $output_dir/$feature_type-$n_cluster-$layer/$split.txt \
        --output $output_dir/$feature_type-$n_cluster-$layer-$mode/$split.txt \
        --mode $mode

done


