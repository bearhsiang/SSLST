#!/usr/local/bin bash

usage() {
    echo "Usage: $0 [flag] [upstream]"
    echo "  -h, --help display this help message"
    echo "  -d, --dataset [dataset]"
    echo "  --split [split]"
    echo "  --audio-key [path]"
    echo "  --audio-root [key]"
    echo "  --model-path [path]"
    echo "  --model-name [name]"
    echo "  --layer [n]"
    echo "  --km-model-path [path]"
    echo "  --suffix [str]"
    echo "  --data-root [root] (default: $data_root)"
    echo "  --tsv-root [root] (default: $tsv_root)"
    echo "  --feat-root [root] (default: $feat_root)"
    echo "  --nshard [N] (default: $nshard)"
    exit 1;
}

OPTIONS=hd:
LONGOPTS=help,dataset:,split:,audio-key:,audio-root:,model-path:,model-name:,layer:,km-model-path:,suffix:,data-root:,tsv-root:,nshard:
PARSED=`getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@"`

eval set -- $PARSED

data_root=data
tsv_root=$data_root/tsv
feat_root=/hdd/ssl_feat
nshard=5
hubert_root=/home/sean/Desktop/SSLST/fairseq/examples/hubert

while true; do
    case "$1" in
        -h | --help ) HELP=true; shift ;;
        --dataset ) dataset=$2; shift 2;;
        --split ) split=$2; shift 2;;
        --audio-key ) audio_key=$2; shift 2;;
        --audio-root ) audio_root=$2; shift 2;;
        --model-path ) model_path=$2; shift 2;;
        --model-name ) model_name=$2; shift 2;;
        --layer ) layer=$2; shift 2;;
        --km-model-path ) km_model_path=$2; shift 2;;
        --suffix ) suffix=$2; shift 2;;
        --data-root ) data_root=$2; shift 2;;
        --tsv-root ) tsv_root=$2; shift 2;;
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
python utils_new/create_fs_manifest.py \
    --audio-root $audio_root \
    --tsv-file $tsv_root/$dataset/$split.tsv \
    --key $audio_key \
    --output $data_root/$dataset/manifest/$split.tsv

### apply km quantization
for rank in `seq 0 $((nshard-1))`; do
    if [ ! -f $feat_root/$dataset/$model_name/$layer/${split}_${rank}_${nshard}.npy ]; then
        python $hubert_root/simple_kmeans/dump_hubert_feature.py \
            $data_root/$dataset/manifest \
            $split \
            $model_path \
            $layer \
            $nshard \
            $rank \
            $feat_root/$dataset/$model_name/$layer
    else
        echo $feat_root/$dataset/$model_name/$layer/${split}_${rank}_${nshard}.npy exist, skip
    fi

    python $hubert_root/simple_kmeans/dump_km_label.py \
        $feat_root/$dataset/$model_name/$layer \
        $split \
        $km_model_path \
        $nshard \
        $rank \
        $data_root/$dataset/$suffix-tmp
done


for rank in `seq 0 $((nshard-1))`; do
    cat $data_root/$dataset/$suffix-tmp/${split}_${rank}_${nshard}.km 
done > $data_root/$dataset/$split.$suffix