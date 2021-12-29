import argparse
from pathlib import Path
import pandas as pd
import csv
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetCreator as KEYS
from tqdm.auto import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input-dir', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-a', '--audio-key', required=True)
    parser.add_argument('-t', '--tgt-key', required=True)
    parser.add_argument('-s', '--src-key')
    parser.add_argument('-i', '--id-key')
    parser.add_argument('-S', '--src-lang-key')
    parser.add_argument('-T', '--tgt-lang-key')
    parser.add_argument('--speaker-key')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.tsv"):
        
        output_file = output_dir/file.name

        if not args.force:
            assert not output_file.exists(), f'output file: {output_file} exists, use -f/--force to force overwrite'
        
        data = pd.read_csv(file, 
            delimiter='\t',
            quoting=csv.QUOTE_NONE,
        )
        
        output = []
        for index, line in tqdm(data.iterrows()):

            item = {}

            if args.id_key:
                ID = line[args.id_key]
            else:
                ID = line[args.audio_key].rsplit('.', maxsplit=1)[0]
            item[KEYS.KEY_ID] = ID

            item[KEYS.KEY_AUDIO] = line[args.audio_key]

            if args.src_key:
                item[KEYS.KEY_SRC_TEXT] = line[args.src_key]
            
            item[KEYS.KEY_TGT_TEXT] = line[args.tgt_key]
            
            if args.src_lang_key:
                item[KEYS.KEY_SRC_LANG] = line[args.src_lang_key]
            
            if args.tgt_lang_key:
                item[KEYS.KEY_TGT_LANG] = line[args.tgt_lang_key]
            
            if args.speaker_key:
                item[KEYS.KEY_SPEAKER] = line[args.speaker_key]

            output.append(item)

        output = pd.DataFrame(output)

        output.to_csv(
            output_file, 
            sep='\t',
            index=False,
            quoting=csv.QUOTE_NONE,
        )