#!/usr/local/bin bash

source script/setup.sh

file=/hdd/NewsCrawl/news.2020.fr.shuffled.deduped

name=newscrawl_fr
max=30
min=10
N=10000

dest_dir=$sslst_data_root/$name

mkdir $dest_dir -p

python utils_new/select_mono_data.py \
    --input $file \
    --max $max \
    --min $min \
    --N $N \
    --output $dest_dir/max$max-min$min-N$N.txt
