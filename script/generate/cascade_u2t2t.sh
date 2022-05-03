#!/usr/local/bin bash

source script/setup.sh

dataset=$1
src_lang=$2
mid_lang=$3
tgt_lang=$4
bpe=8000

u2t_arch=transformer_iwslt_de_en
u2t_ckpt=$sslst_output_root/$dataset/$u2t_arch-$src_lang-$mid_lang/checkpoint_best.pt

t2t_arch=transformer_iwslt_de_en
t2t_ckpt=$sslst_output_root/$dataset/$t2t_arch-$mid_lang-$tgt_lang/checkpoint_best.pt

tmp_dir=$sslst_data_root/$dataset/$src_lang-$mid_lang-$tgt_lang
mkdir -p $tmp_dir

fairseq-generate \
    $sslst_data_bin_root/$dataset/$src_lang-$mid_lang \
    --gen-subset test \
    --path $u2t_ckpt \
    --beam 5 \
    --max-len-a 0 \
    --max-len-b 256 \
    --max-source-positions 4096 \
    --remove-bpe sentencepiece \
    --max-tokens 8192 \
    --scoring wer 2>&1 | tee $tmp_dir/u2t.out

cat $tmp_dir/u2t.out | grep ^S | LC_ALL=C sort -V | cut -f2- > $tmp_dir/u2t.$src_lang
cat $tmp_dir/u2t.out | grep ^D | LC_ALL=C sort -V | cut -f3- > $tmp_dir/u2t.$mid_lang

spm_model=$sslst_data_root/$dataset/tmp/$mid_lang-$bpe.model

spm_encode \
    --model=$spm_model \
    --output_format=piece \
    < $tmp_dir/u2t.$mid_lang \
    > $tmp_dir/u2t.spm.$mid_lang

cp $sslst_data_root/$dataset/test.$tgt_lang $tmp_dir/u2t.spm.$tgt_lang

fairseq-preprocess \
    -s $mid_lang \
    -t $tgt_lang \
    --testpref $tmp_dir/u2t.spm \
    --destdir $tmp_dir/data-bin \
    --srcdict $sslst_data_bin_root/$dataset/$mid_lang-$tgt_lang/dict.$mid_lang.txt \
    --tgtdict $sslst_data_bin_root/$dataset/$mid_lang-$tgt_lang/dict.$tgt_lang.txt \
    --workers 4

fairseq-generate \
    $tmp_dir/data-bin \
    --gen-subset test \
    --path $t2t_ckpt \
    --beam 5 \
    --max-len-a 1.5 \
    --max-len-b 20 \
    --scoring sacrebleu \
    --remove-bpe sentencepiece \
    --max-tokens 8192