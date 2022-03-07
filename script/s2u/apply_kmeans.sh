#!/usr/local/bin bash

source script/setup.sh

usage() {
    echo "Usage: $0 [flag] [upstream]"
    echo "  -h, --help display this help message"
    echo "  -d, --dataset [dataset]"
    echo "  --split [split]"
    echo "  --audio-key [path]"
    echo "  --audio-root [key]"
    echo "  --model [name]"
    echo "  --layer [n]"
    echo "  --km-model-path [path]"
    echo "  --suffix [str]"
    echo "  --nshard [N] (default: $nshard)"
    exit 1;
}

OPTIONS=hd:
LONGOPTS=help,dataset:,split:,audio-key:,audio-root:,model:,layer:,km-model-path:,suffix:,nshard:
PARSED=`getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@"`

eval set -- $PARSED

nshard=5
hubert_root=$sslst_fairseq_root/examples/hubert

while true; do
    case "$1" in
        -h | --help ) HELP=true; shift ;;
        --dataset ) dataset=$2; shift 2;;
        --split ) split=$2; shift 2;;
        --audio-key ) audio_key=$2; shift 2;;
        --audio-root ) audio_root=$2; shift 2;;
        --model ) model=$2; shift 2;;
        --layer ) layer=$2; shift 2;;
        --km-model-path ) km_model_path=$2; shift 2;;
        --suffix ) suffix=$2; shift 2;;
        --nshard ) nshard=$2; shift 2;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

if [ ! -z $HELP ]; then
    usage
    exit 1;
fi

### create manifest
if [ ! -f $sslst_data_root/$dataset/manifest/$split.tsv ]; then
    python utils_new/create_fs_manifest.py \
        --audio-root $audio_root \
        --tsv-file $sslst_data_root/tsv/$dataset/$split.tsv \
        --key $audio_key \
        --output $sslst_data_root/$dataset/manifest/$split.tsv
else
    echo "$sslst_data_root/$dataset/manifest/$split.tsv exists, skip..."
fi

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
            --output-dir $sslst_feat_root/$dataset/$model/$layer
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