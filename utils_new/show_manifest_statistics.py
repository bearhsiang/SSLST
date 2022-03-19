import argparse
import datetime

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-file')
    parser.add_argument('--sample-rate', type=int, default=16000)
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
        
        print('audio root:', audio_root)
        print('# of raws:', len(audio_files))
        total_frame = sum(n_frames)
        print('total frames:', total_frame)
        print(f'duration: {total_frame/args.sample_rate/3600:.2f} hr')

if __name__ == '__main__':

    args = get_args()
    main(args)