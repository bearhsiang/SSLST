data_root=data
origin_dataset=must-c_en_de-s2u_cpc_50_2
output_dataset=$origin_dataset-reduce

src_lang=cpc-50
tgt_lang=spm.de

mkdir -p $data_root/$output_dataset

for split in train dev tst-COMMON tst-HE; do
    python fairseq_s2u/reduce.py \
        -i $data_root/$origin_dataset/$split.$src_lang \
        -o $data_root/$output_dataset/$split.$src_lang \

    cp $data_root/$origin_dataset/$split.$tgt_lang $data_root/$output_dataset    
done

fairseq-preprocess \
    -s $src_lang \
    -t $tgt_lang \
    --trainpref $data_root/$output_dataset/train \
    --validpref $data_root/$output_dataset/dev \
    --testpref $data_root/$output_dataset/tst-HE,$data_root/$output_dataset/tst-COMMON \
    --destdir data-bin/$output_dataset