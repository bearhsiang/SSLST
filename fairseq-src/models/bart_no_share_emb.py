from fairseq.models.bart.model import mbart_large_architecture
from fairseq.models.transformer.transformer_legacy import transformer_iwslt_de_en
from fairseq.models import register_model_architecture

@register_model_architecture("bart", "mbart_large_no_share_emb")
def mbart_large_no_share_emb_architecture(args):
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    mbart_large_architecture(args)

@register_model_architecture("transformer", "transformer_iwslt_de_en_mbart_large")
def iwslt_mbart_large(args):

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)

    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    transformer_iwslt_de_en(args)