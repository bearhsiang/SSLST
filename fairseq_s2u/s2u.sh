fairseq_root=fairseq

data_root=data
dataset=must-c_en_de
wav_dir=/hdd/must-c_en_de-wav


AUDIO_KEY=audio
TEXT_KEY=tgt_text
TGT_LANG=de

type=hubert
n_cluster=200
km_model_dir=data/km_models
ac_model_dir=data/ac_models
layer=-1
N=1000

out_dataset=${dataset}-s2u_${type}_${n_cluster}_$layer

mkdir -p $data_root/$out_dataset/tmp

for split in train dev tst-COMMON tst-HE; do

    mkdir -p $data_root/$out_dataset/tmp/manifest/$split

    python utils/create_s2u_manifest.py \
        --audio-root $wav_dir \
        --tsv-file $data_root/$dataset/$split.tsv \
        --key $AUDIO_KEY \
        --output-prefix $data_root/$out_dataset/tmp/manifest/$split/ \
        -N $N
    
    mkdir -p $data_root/$out_dataset/tmp/quantized/$split

    for file in `ls $data_root/$out_dataset/tmp/manifest/$split`; do 
        
        output_file=$data_root/$out_dataset/tmp/quantized/$split/$file

        if [ -f $output_file ]; then
            echo "$output_file exist!, skip"
            continue
        fi

        python $fairseq_root/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
            --feature_type $type \
            --kmeans_model_path $km_model_dir/$type-$n_cluster.bin \
            --layer $layer \
            --acoustic_model_path $ac_model_dir/$type.pt \
            --manifest_path $data_root/$out_dataset/tmp/manifest/$split/$file \
            --out_quantized_file_path $output_file \
            --extension .wav
    done

    python fairseq_s2u/create_data.py \
        --tsv $data_root/$dataset/$split.tsv \
        --audio-key $AUDIO_KEY \
        --text-key $TEXT_KEY \
        --lang $TGT_LANG \
        --quant-dir $data_root/$out_dataset/tmp/quantized/$split \
        --quant-lang $type-$n_cluster \
        --output-prefix $data_root/$out_dataset/$split

done

spm_train \
    --input=$data_root/$out_dataset/train.$TGT_LANG \
    --model_prefix $data_root/$out_dataset/spm-$TGT_LANG \
    --character_coverage 1 \
    --model_type char

for split in train dev tst-COMMON tst-HE; do
    spm_encode \
        --model $data_root/$out_dataset/spm-$TGT_LANG.model \
        < $data_root/$out_dataset/$split.$TGT_LANG \
        > $data_root/$out_dataset/$split.spm.$TGT_LANG
done

fairseq-preprocess \
    -s $type-$n_cluster \
    -t spm.$TGT_LANG \
    --trainpref $data_root/$out_dataset/train \
    --validpref $data_root/$out_dataset/dev \
    --testpref $data_root/$out_dataset/tst-HE,$data_root/$out_dataset/tst-COMMON \
    --destdir data-bin/$out_dataset