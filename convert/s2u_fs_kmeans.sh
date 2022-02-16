#!/usr/local/bin bash

hubert_root=/home/sean/Desktop/SSLST/fairseq/examples/hubert
tsv_dir=data/s2u/libritrans-en-fr/manifest_fs
ckpt_path=data/ssl_models/hubert_base_ls960.pt
km_path=data/kmeans_model/hubert_base_ls960_L9_km500.bin
feat_dir=/hdd/ssl_feat
layer=9
nshard=5
rank=0
lab_dir=data/s2u/libritrans-en-fr/hubert-l9-km500

for split in train dev test; do
    # for rank in `seq 0 $((nshard-1))`; do

    #     python $hubert_root/simple_kmeans/dump_hubert_feature.py \
    #         $tsv_dir \
    #         $split \
    #         $ckpt_path \
    #         $layer \
    #         $nshard \
    #         $rank \
    #         $feat_dir

    #     python $hubert_root/simple_kmeans/dump_km_label.py \
    #         $feat_dir \
    #         $split \
    #         $km_path \
    #         $nshard \
    #         $rank \
    #         $lab_dir

    # done

    for rank in `seq 0 $((nshard-1))`; do

        cat $lab_dir/${split}_${rank}_${nshard}.km >> $lab_dir/$split.km

    done

done
