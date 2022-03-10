from collections import OrderedDict
from fairseq.models import register_model_architecture, register_model
from fairseq.models.bart import mbart_large_architecture, bart_base_architecture
from fairseq.models.speech_to_text.s2t_transformer import base_architecture, S2TTransformerModel
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.modules.multihead_attention import MultiheadAttention

@register_model("s2t_transformer_pt")
class S2TTransformerModel_pt(S2TTransformerModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--pt-encoder-path')
        parser.add_argument('--pt-encoder-arch')
        parser.add_argument('--pt-decoder-path')
        parser.add_argument('--pt-decoder-arch')
        S2TTransformerModel.add_args(parser)

    @classmethod
    def build_encoder(cls, args):
        encoder = super().build_encoder(args)
        pt_encoder_path = getattr(args, 'pt_encoder_path', None)
        if pt_encoder_path:
            pt_encoder_arch = getattr(args, 'pt_encoder_arch', None)
            assert pt_encoder_arch, 'you have to provide the architecture of pretrained encoder'
            load_state = OrderedDict()
            if pt_encoder_arch in ['mbart_large']:
                pt_state = load_checkpoint_to_cpu(pt_encoder_path)
                self_attn_proj = [f'self_attn.{p}_proj' for p in ['q', 'k', 'v', 'out']]
                other = ['fc1', 'fc2']
                for l in range(args.encoder_layers):
                    MultiheadAttention.upgrade_state_dict_named(None, pt_state['model'], f'encoder.layers.{l}.self_attn')
                    for w in ['weight', 'bias']:
                        for m in self_attn_proj + other:
                            load_state[f'transformer_layers.{l}.{m}.{w}'] = pt_state['model'][f'encoder.layers.{l}.{m}.{w}']
                        load_state[f'transformer_layers.{l}.self_attn_layer_norm.{w}'] = pt_state['model'][f'encoder.layers.{l}.layer_norms.0.{w}']
                        load_state[f'transformer_layers.{l}.final_layer_norm.{w}'] = pt_state['model'][f'encoder.layers.{l}.layer_norms.1.{w}']
            elif pt_encoder_arch in ['bart_base']:
                pt_state = load_checkpoint_to_cpu(pt_encoder_path)
                for k in pt_state['model'].keys():
                    if k.startswith('encoder.layers'):
                        load_state[k.replace('encoder.', 'transformer_')] = pt_state['model'][k]
            else:
                raise NotImplementedError

            encoder_state_dict = encoder.state_dict()
            for k in load_state:
                assert k in encoder_state_dict
            encoder.load_state_dict(load_state, strict=False)

        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = super().build_decoder(args, task, embed_tokens)
        pt_decoder_path = getattr(args, 'pt_decoder_path', None)
        if pt_decoder_path:
            pt_decoder_arch = getattr(args, 'pt_decoder_arch', None)
            assert pt_decoder_arch, 'you have to provide the architecture of pretrained decoder'
            load_state = OrderedDict()
            if pt_decoder_arch in ['mbart_large', 'bart_base']:
                pt_state = load_checkpoint_to_cpu(pt_decoder_path)
                # modules to load
                self_attn_proj = [f'self_attn.{p}_proj' for p in ['q', 'k', 'v', 'out']]
                self_attn_ln = ['self_attn_layer_norm']
                encoder_attn_proj = [f'encoder_attn.{p}_proj' for p in ['q', 'k', 'v', 'out']]
                encoder_attn_ln = ['encoder_attn_layer_norm']
                other = ['fc1', 'fc2', 'final_layer_norm']
                for l in range(args.decoder_layers):
                    MultiheadAttention.upgrade_state_dict_named(None, pt_state['model'], f'decoder.layers.{l}.self_attn')
                    MultiheadAttention.upgrade_state_dict_named(None, pt_state['model'], f'decoder.layers.{l}.encoder_attn')
                    for w in ['weight', 'bias']:
                        for m in self_attn_proj + self_attn_ln + encoder_attn_proj + encoder_attn_ln + other:
                            load_state[f'layers.{l}.{m}.{w}'] = pt_state['model'][f'decoder.layers.{l}.{m}.{w}']
            else:
                raise NotImplementedError

            decoder_state_dict = decoder.state_dict()
            for k in load_state:
                assert k in decoder_state_dict
            decoder.load_state_dict(load_state, strict=False)

        return decoder

@register_model_architecture(model_name="s2t_transformer_pt", arch_name="s2t_transformer_mbart_large")
def s2t_transformer_mbart_large(args):
    mbart_large_architecture(args)
    base_architecture(args)

@register_model_architecture(model_name='s2t_transformer_pt', arch_name='s2t_transformer_bart_base')
def s2t_transformer_bart_base(args):
    bart_base_architecture(args)
    base_architecture(args)
