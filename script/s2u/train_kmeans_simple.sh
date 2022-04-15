#!/usr/local/bin bash

source script/setup.sh

dataset=$1
dataset_km_tag=$2
model=$3
layer=$4
n_cluster=$5
percent=$6

split=train
nshard=5
format=collect
device=cuda

hubert_root=$sslst_fairseq_root/examples/hubert
feat_dir=$sslst_feat_root/$dataset/$model/$layer

km_path=$sslst_data_root/kmeans_model/$model-${dataset_km_tag}${percent}p-L$layer-km$n_cluster.bin

if [ -f $km_path ]; then 
    echo km model: $km_path exists
    exit
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

python $hubert_root/simple_kmeans/learn_kmeans.py \
    $feat_dir \
    $split \
    $nshard \
    $km_path \
    $n_cluster \
    --seed $sslst_seed \
    --percent $percent