import argparse
from tempfile import NamedTemporaryFile
from data_utils import read_tsv
import sentencepiece as sp
import shutil
from pathlib import Path

# fairseq's special token
UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1

def create_sentencepiece(filenames, model_type, vocab_size, output_prefix):

    sp.SentencePieceTrainer.train(
        input=','.join(filenames),
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        unk_id=UNK_TOKEN_ID,
        bos_id=BOS_TOKEN_ID,
        eos_id=EOS_TOKEN_ID,
        pad_id=PAD_TOKEN_ID,
    )

    spm = sp.SentencePieceProcessor(
        model_file=f'{output_prefix}.model'
    )

    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}

    assert vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
    assert vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
    assert vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    assert vocab.get(PAD_TOKEN_ID) == PAD_TOKEN

    vocab = {
        i: s for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }

    with open(f'{output_prefix}.txt', 'w') as f:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            print(f'{s} 1', file=f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', required=True)
    parser.add_argument('-p', '--pattern', default='train*.tsv')
    parser.add_argument('-k', '--keys', default='tgt_text')
    parser.add_argument('-o', '--output-prefix')
    parser.add_argument('-n', '--vocab-size', default=1000)
    parser.add_argument('--model-type', default='char')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    keys = args.keys.split(',')

    with NamedTemporaryFile(mode='w') as temp_file: 
        
        for file in input_dir.glob(args.pattern):
            lines = read_tsv(file)
            for line in lines:
                for key in keys:
                    print(line[key], file=temp_file)
        
        temp_file.flush()

        create_sentencepiece(
            [temp_file.name],
            args.model_type,
            args.vocab_size,
            args.output_prefix,
        )

