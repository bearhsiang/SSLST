import argparse
from pathlib import Path
import torchaudio
from tqdm.auto import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir')
    parser.add_argument('-o', '--output-dir')
    parser.add_argument('-r', '--rate', type=float, default=16000)
    args = parser.parse_args()

    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)

    args.output_dir.mkdir(exist_ok=True)

    resamplers = {}

    for file in tqdm(args.input_dir.glob('*.mp3')):
        
        stem = file.stem
        output_file = args.output_dir/f'{stem}.wav'

        if output_file.exists():
            print(output_file, 'exists, skip...')
            continue

        source, sr = torchaudio.load(file)

        source = torchaudio.functional.resample(
            source, 
            orig_freq = sr,
            new_freq = args.rate,
        )

        torchaudio.save(output_file, source, args.rate)
        
