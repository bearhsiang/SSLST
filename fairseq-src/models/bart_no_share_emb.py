from fairseq.models.bart.model import mbart_large_architecture
from fairseq.models import register_model_architecture

@register_model_architecture("bart", "mbart_large_no_share_emb")
def mbart_large_no_share_emb_architecture(args):
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    mbart_large_architecture(args)