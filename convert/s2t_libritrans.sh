#!/usr/local/bin bash

# config

input_root=data/tsv/
output_root=data/s2t/

name=libritrans-en-fr

### 

input_dir=$input_root/$name
output_dir=$output_root/$name/raw

python convert/s2t.py \
    -d $input_dir \
    -o $output_dir \
    -a audio \
    -t tgt_text \
    -s src_text \