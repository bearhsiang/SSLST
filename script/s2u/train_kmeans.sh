#!/usr/local/bin bash

source script/setup.sh

dataset=covost2-de-en
dataset_root=$sslst_covost2_data_root
dataset_km_tag=cvde
split=train
key=path

hubert_root=$sslst_fairseq_root/examples/hubert
model=$1
layer=$2
n_cluster=$3
nshard=5
format=collect
device=cuda

feat_dir=$sslst_feat_root/$dataset/$model/$layer

### create manifest
echo "create manifest"
if [ ! -f "$sslst_data_root/$dataset/manifest/$split.tsv" ]; then
    python utils_new/create_fs_manifest.py \
        --audio-root $dataset_root \
        --tsv-file $sslst_data_root/tsv/$dataset/$split.tsv \
        --key $key \
        --output $sslst_data_root/$dataset/manifest/$split.tsv
else
    echo "$sslst_data_root/$dataset/manifest/$split.tsv exists, skip"
fi

### dump features
for rank in `seq 0 $((nshard-1))`; do
    if [ -f $feat_dir/${split}_${rank}_${nshard}.npy ] && [ -f $feat_dir/${split}_${rank}_${nshard}.len ]; then
        echo $feat_dir/${split}_${rank}_${shard}.[npy,len] exist, skip
    else
        python utils_new/dump_features.py \
            --manifest $sslst_data_root/$dataset/manifest \
            --split $split \
            --model $model \
            --layer $layer \
            --nshard $nshard \
            --rank $rank \
            --device $device \
            --output-dir $sslst_feat_root/$dataset/$model/$layer \
            --format $format
    fi
done

km_path=$sslst_data_root/kmeans_model/$model-$dataset_km_tag-L$layer-km$n_cluster.bin

python $hubert_root/simple_kmeans/learn_kmeans.py \
    $feat_dir \
    $split \
    $nshard \
    $km_path \
    $n_cluster
    # --percent 0.1