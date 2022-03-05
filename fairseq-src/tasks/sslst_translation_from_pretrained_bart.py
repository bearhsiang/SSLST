from fairseq.tasks.translation_from_pretrained_bart import TranslationFromPretrainedBARTTask
from fairseq.tasks import register_task

@register_task("sslst_translation_from_pretrained_bart")
class sslst_TranslationFromPretrainedBARTTask(TranslationFromPretrainedBARTTask):

    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.args = cfg