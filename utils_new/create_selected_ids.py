import argparse
import pandas as pd

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-file', required=True)
    parser.add_argument('--sample-rate', type=int, default=16_000)
    parser.add_argument('--proportion', type=float)
    parser.add_argument('--hour', type=float)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    return args

def main(args):

    with open(args.manifest_file) as f:
        audio_root = f.readline().strip()
        audio_files, n_frames = [], []
        for line in f:
            audio_file, n_frame = line.strip().split('\t')
            audio_files.append(audio_file)
            n_frames.append(int(n_frame))

    df = pd.DataFrame({
        'audio_file': audio_files,
        'n_frame': n_frames,
    })
    df = df.sample(frac=1, random_state=args.seed)
    if args.hour:
        df['cum_frames'] = df['n_frame'].cumsum()
        select_ids = df.index[df['cum_frames'] < (args.hour * 3600 * args.sample_rate)]
    else:
        assert args.proportion, "you have to set one of 'time' or 'proportion'"
        select_ids = df.index[:round(args.proportion*len(df))]

    with open(args.output, 'w') as f:
        for index in sorted(select_ids):
            print(index, file=f)


if __name__ == '__main__':

    args = get_args()
    main(args)