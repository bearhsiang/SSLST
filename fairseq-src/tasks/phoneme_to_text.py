from fairseq.tasks.translation import TranslationTask, TranslationConfig
from fairseq.tasks import register_task

@register_task("phoneme_to_text", dataclass=TranslationConfig)
class PhonemeToText(TranslationTask):

    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

