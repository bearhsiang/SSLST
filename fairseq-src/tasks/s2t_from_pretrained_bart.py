
from argparse import Action
from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.tasks import register_task

@register_task('speech_to_text_from_pretrained_bart')
class SpeechToTextFromPretrainedBart(SpeechToTextTask):

    @classmethod
    def add_args(cls, parser):
        SpeechToTextTask.add_args(parser)

        parser.add_argument('--langs', type=str)
        parser.add_argument('--prepend-bos', action='store_true')

    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        self.langs = args.langs.split(',')
        for l in self.langs:
            self.tgt_dict.add_symbol(f'<lang:{l}>')
        self.tgt_dict.add_symbol('<mask>')
        if args.prepend_bos:
            self.data_cfg.config['prepend_tgt_lang_tag'] = True
    



