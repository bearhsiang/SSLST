import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-file')
    parser.add_argument('--text-file')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    return args

def main(args):
    output_dir = Path(args.output_dir)
    with open(args.manifest_file, 'r') as manifest_f:
        manifest_f.readline()
        with open(args.text_file, 'r') as text_f:
            for line in manifest_f:
                wav_path, _ = line.strip().split('\t')
                index = Path(wav_path).stem
                text = text_f.readline().strip()
                print(text, file=open(output_dir/f'{index}.hu', 'w'))

if __name__ == '__main__':
    args = get_args()
    main(args)