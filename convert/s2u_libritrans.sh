#!/usr/local/bin bash

# config
root=data
name=libritrans-en-fr
input_dir=data/s2t/$name/normalized
output_dir=data/s2u/$name
audio_root=/hdd/libritrans
audio_key=audio

### config for quantization
fairseq_root=fairseq
feature_type=logmel
n_cluster=50
km_models_dir=data/km_models
ac_models_dir=data/ac_models
layer=-1
ext=.wav

### config for reduction
mode=simple

### 

for split in test dev train; do

    # python convert/s2u_manifest.py \
    #     --audio-root $audio_root/$split/audiofiles \
    #     --tsv-file $input_dir/$split.tsv \
    #     --key $audio_key \
    #     --output $output_dir/manifest/$split.txt

    # python $fairseq_root/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    #     --feature_type $feature_type \
    #     --kmeans_model_path $km_models_dir/$feature_type-$n_cluster.bin \
    #     --acoustic_model_path $ac_models_dir/$feature_type.pt \
    #     --layer $layer \
    #     --manifest_path $output_dir/manifest/$split.txt \
    #     --out_quantized_file_path $output_dir/quantized-$feature_type-$n_cluster/$split.txt \
    #     --extension $ext

    python convert/s2u_reduce.py \
        --input $output_dir/quantized-$feature_type-$n_cluster/$split.txt \
        --output $output_dir/quantized-$feature_type-$n_cluster-$mode/$split.txt \
        --mode $mode

done


