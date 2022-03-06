import argparse
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--dict')
    parser.add_argument('--output')
    parser.add_argument('--mode', default='random', choices=['random', 'max', 'min'])
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    return args

def main(args):
    
    random.seed(args.seed)

    input_record = {}

    for line in open(args.input, 'r'):
        for w in line.strip().split():
            if w in input_record:
                input_record[w] += 1
            else:
                input_record[w] = 1

    dict_record = {}

    for line in open(args.dict, 'rb'):
        line = line.split()
        w, n = line[0].decode(), line[1].decode()
        dict_record[w] = float(n)

    sorted_input = sorted(input_record.keys(), key=lambda w: input_record[w], reverse=True)
    sorted_dict = sorted(dict_record.keys(), key=lambda w: dict_record[w], reverse=True)
    
    if args.mode == 'random':
        random.shuffle(sorted_dict)
    elif args.mode == 'max':
        pass
    elif args.mode == 'min':
        sorted_dict.reverse()
    
    map_list = sorted_dict
    map_list = map_list[:len(sorted_input)]

    with open(args.output, 'w') as f:
        for src, tgt in zip(sorted_input, map_list):
            print(src, tgt, file=f)

if __name__ == '__main__':
    args = get_args()
    main(args)