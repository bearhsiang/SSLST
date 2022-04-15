#!/usr/local/bin bash

source script/setup.sh

dataset=$1
model=$2
layer=$3
n_cluster=$4
km_tag=$5

splits="train dev test"
nshard=5

km_model_path=$sslst_data_root/kmeans_model/$model-$km_tag-L$layer-km$n_cluster.bin
suffix=${model}_l${layer}_${km_tag}${n_cluster}
hubert_root=$sslst_fairseq_root/examples/hubert

for split in $splits; do

    ### apply km quantization
    for rank in `seq 0 $((nshard-1))`; do
        if [ ! -f $sslst_feat_root/$dataset/$model/$layer/${split}_${rank}_${nshard}.npy ]; then
            python utils_new/dump_features.py \
                --manifest-dir $sslst_data_root/$dataset/manifest \
                --split $split \
                --model $model \
                --layer $layer \
                --nshard $nshard \
                --rank $rank \
                --output-dir $sslst_feat_root/$dataset/$model/$layer \
                --sample-rate $sslst_sample_rate
        else
            echo $sslst_feat_root/$dataset/$model/$layer/${split}_${rank}_${nshard}.npy exist, skip
        fi

        python $hubert_root/simple_kmeans/dump_km_label.py \
            $sslst_feat_root/$dataset/$model/$layer \
            $split \
            $km_model_path \
            $nshard \
            $rank \
            $sslst_data_root/$dataset/$suffix-tmp
    done


    for rank in `seq 0 $((nshard-1))`; do
        cat $sslst_data_root/$dataset/$suffix-tmp/${split}_${rank}_${nshard}.km 
    done > $sslst_data_root/$dataset/$split.$suffix

done