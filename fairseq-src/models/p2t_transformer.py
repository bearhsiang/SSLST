from fairseq.models.speech_to_text.s2t_transformer import S2TTransformerModel, S2TTransformerEncoder
import fairseq.models.speech_to_text.s2t_transformer as s2t
from fairseq.models.transformer import Embedding
from fairseq.models import register_model, register_model_architecture
from typing import Dict, List, Optional, Tuple
import torch
from fairseq.modules import LayerNorm

@register_model("p2t_transformer")
class P2TTransformerModel(S2TTransformerModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--input-feat-per-channel', type=int)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        encoder_embed_tokens = build_embedding(
            task.source_dictionary, args.input_feat_per_channel,
        )

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )

        args.input_channels = 1

        encoder = cls.build_encoder(args, task, encoder_embed_tokens)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, task, encoder_embed_tokens):
        return P2TTransformerEncoder(args, task.source_dictionary, encoder_embed_tokens)

class P2TTransformerEncoder(S2TTransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args)

        self.embed_tokens = embed_tokens
        self.padding_idx = embed_tokens.padding_idx
        
        if args.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(args.input_feat_per_channel, export=args.export)
        else:
            self.layernorm_embedding = None

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        # don't add pos emb before subsample
        # if self.embed_positions is not None:
        #     x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings
        )

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return super().forward(x, src_lengths, return_all_hiddens)

@register_model_architecture(model_name="p2t_transformer", arch_name="p2t_transformer")
def base_architecture(args):
    s2t.base_architecture(args)
    # transformer's config
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.export = getattr(args, "export", False)

@register_model_architecture("p2t_transformer", "p2t_transformer_s")
def p2t_transformer_s(args):
    args.input_feat_per_channel = getattr(args, 'input_feat_per_channel', 256)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("p2t_transformer", "p2t_transformer_s_enc_6")
def p2t_transformer_s_enc_6(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    p2t_transformer_s(args)

@register_model_architecture("p2t_transformer", "p2t_transformer_s_enc_0")
def p2t_transformer_s_enc_0(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 0)
    p2t_transformer_s(args)
