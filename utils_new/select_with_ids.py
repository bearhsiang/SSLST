import argparse
from pathlib import Path

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--selected-ids-file')
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--has-header', action='store_true')
    args = parser.parse_args()

    return args

def main(args):

    ids = set([int(line.strip()) for line in open(args.selected_ids_file)])

    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)

    with open(args.output, 'w') as out_f:
        with open(args.input, 'r') as in_f:
            if args.has_header:
                header = in_f.readline().strip()
                print(header, file=out_f)
            for id, line in enumerate(in_f):
                if int(id) in ids:
                    print(line.strip(), file=out_f)

    
if __name__ == '__main__':

    args = get_args()
    main(args)