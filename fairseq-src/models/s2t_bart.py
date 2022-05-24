from fairseq.models.bart.model import BARTModel, mbart_large_architecture
from fairseq.models import register_model, register_model_architecture
import torch

@register_model("s2t_bart")
class S2TBART(BARTModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):

        B, N, D = src_tokens.shape
        fake_src_tokens = src_tokens.new_full((B, N), self.encoder.padding_idx, dtype=torch.long)
        for i, length in enumerate(src_lengths):
            fake_src_tokens[i, :length] = -1

        print(fake_src_tokens, src_lengths)

        # raise

        return super().forward(
            fake_src_tokens, 
            src_lengths, 
            prev_output_tokens,
            token_embeddings = src_tokens,
        )

@register_model_architecture('s2t_bart', 's2t_mbart_large')
def s2t_mbart_large_architecture(args):
    mbart_large_architecture(args)
