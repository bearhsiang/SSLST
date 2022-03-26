import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--max', type=int, default=-1)
    parser.add_argument('--min', type=int, default=-1)
    parser.add_argument('--N', type=int, default=-1)
    parser.add_argument('--spm-model')
    parser.add_argument('--output')
    args = parser.parse_args()

    return args

def default_tokenizer():

    def f(s):
        return s.split()

    return f

def spm_tokenizer(model_file):

    import sentencepiece as spm
    model = spm.SentencePieceProcessor(model_file=model_file)

    def f(s):
        return model.encode(s, out_type=str)

    return f 

def main(args):

    tokenizer = default_tokenizer() if not args.spm_model else spm_tokenizer(args.spm_model)

    count = 0

    with open(args.output, 'w') as f:
        for line in open(args.input, 'r'):
            line = line.strip()
            s = tokenizer(line)
            if args.max > 0 and len(s) > args.max:
                continue
            if args.min > 0 and len(s) < args.min:
                continue
            print(line, file=f)
            count += 1
            if args.N > 0 and count >= args.N:
                break

if __name__ == '__main__':

    args = get_args()
    main(args)