import argparse
from data_utils import read_tsv
from pathlib import Path
from tqdm.auto import tqdm
import tempfile
import shutil
import subprocess

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-tsv', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('--temp-dir', default='/tmp/mfa-align')
    parser.add_argument('--dict', required=True)
    parser.add_argument('--lang', required=True)
    parser.add_argument('--audio-dir', required=True)
    parser.add_argument('--audio-key', default='audio')
    parser.add_argument('--text-key', default='src_text')
    parser.add_argument('--batch-size', default=1000, type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = Path(args.audio_dir)
    temp_dir = Path(args.temp_dir)
    assert not temp_dir.exists(), f'temp dir: "{temp_dir}" exists.'

    temp_dir.mkdir(parents=True, exist_ok=True)
    lines = read_tsv(args.input_tsv)

    count = 0

    for line in tqdm(lines):

        audio = line[args.audio_key]
        stem = audio.rsplit('.', maxsplit=1)[0]

        # check alignment exists or not
        if (output_dir/f'{stem}.TextGrid').exists():
            # print(stem, 'exist')
            continue

        # move audio to temp_dir
        shutil.copy(audio_dir/audio, temp_dir)

        # create text in temp_dir
        text = line[args.text_key]
        with open(temp_dir/f'{stem}.lab', 'w') as f:
            print(text.upper().replace('-', ''), file=f)
        
        count += 1

        if count == args.batch_size:
            # run alignment
            subprocess.run(['mfa', 'align', temp_dir, args.dict, args.lang, output_dir, '-c'])
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            count = 0

    if count != 0:

        subprocess.run(['mfa', 'align', temp_dir, args.dict, args.lang, output_dir, '-c'])
    
    shutil.rmtree(temp_dir)


    