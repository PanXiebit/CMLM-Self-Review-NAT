

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import (
    FairseqDecoder, FairseqEncoder, FairseqLanguageModel, BaseFairseqModel,
    register_model, register_model_architecture, FairseqEncoderDecoderGanModel,
    FairseqIncrementalDecoder, FairseqModel
)

from fairseq import options
from fairseq import utils
from .bert_seq2seq import TransformerEncoder, SelfTransformerDecoder, base_architecture

from fairseq.modules import (
    AdaptiveSoftmax, CharacterTokenEmbedder, MultiheadAttention,
    SimpleSinusoidalPositionalEmbedding, LearnedPositionalEmbedding
)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx):
    m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx)
    nn.init.normal_(m.weight, mean=0, std=0.02)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

@register_model('bert_transformer_seq2seq_gan')
class Transformer_nonautoregressive_gan(FairseqEncoderDecoderGanModel):
    def __init__(self, args, encoder, decoder, decoder_embed_tokens, tgt_dict):
        super().__init__(args, encoder, decoder, decoder_embed_tokens, tgt_dict)
        print("Model setting Right here!")
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        #for ds in task.datasets.values():
        #    ds.target_is_source = True

        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, is_encoder, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, is_encoder=True, path=args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, is_encoder=True, path=args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, is_encoder=False, path=args.decoder_embed_path
            )


        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, args.encoder_embed_scale)
        decoder = SelfTransformerDecoder(args, tgt_dict, decoder_embed_tokens, args.decoder_embed_scale, remove_head=True)
        return Transformer_nonautoregressive_gan(args, encoder, decoder, decoder_embed_tokens, tgt_dict)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-enc-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--no-dec-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--embedding-only', default=False, action='store_true',
                            help='if set, replaces the encoder with just token embeddings (could be complex e.g. bilm')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--share_gen_dec_embedding', action='store_true',
                            help='share generator, discriminator embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--share_gen_dec_all_weights', action='store_true',
                            help='share generator, discriminator all weights'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--share_gen_dec_encoder_weights', action='store_true',
                            help='share generator, discriminator all weights'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--dis_weight', type=float, default=50.0,
                            help='the coefficent of discriminator loss'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--temperature', type=float, default=1.0,
                            help='the coefficent of gumbel softmax')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--bilm-model-dropout', default=0.1, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the model dropout for bilm')
        parser.add_argument('--bilm-attention-dropout', default=0.0, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the attention dropout for bilm')
        parser.add_argument('--bilm-relu-dropout', default=0.0, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the relu dropout for bilm')
        parser.add_argument('--bilm-mask-last-state', action='store_true',
                            help='if set, masks last state in bilm as is done during training')
        parser.add_argument('--bilm-add-bos', action='store_true',
                            help='if set, adds bos to input')
        parser.add_argument('--decoder-embed-scale', type=float,
                            help='scaling factor for embeddings used in decoder')
        parser.add_argument('--encoder-embed-scale', type=float,
                            help='scaling factor for embeddings used in encoder')



@register_model_architecture('bert_transformer_seq2seq_gan', 'bert_transformer_seq2seq_gan')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', args.encoder_embed_dim * 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', args.encoder_embed_dim // 64)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', args.encoder_attention_heads)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.share_gen_dec_embedding = getattr(args, 'share_gen_dec_embedding', False)
    args.share_gen_dec_all_weights = getattr(args, 'share_gen_dec_all_weights', False)
    args.share_gen_dec_encoder_weights = getattr(args, 'share_gen_dec_encoder_weights', False)
    args.dis_weight = getattr(args, 'dis_weight', 50.0)
    args.temperature = getattr(args, 'temperature', 1.0)

    args.no_enc_token_positional_embeddings = getattr(args, 'no_enc_token_positional_embeddings', False)
    args.no_dec_token_positional_embeddings = getattr(args, 'no_dec_token_positional_embeddings', False)
    args.embedding_only = getattr(args, 'embedding_only', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.decoder_embed_scale = getattr(args, 'decoder_embed_scale', None)
    args.encoder_embed_scale = getattr(args, 'encoder_embed_scale', None)

    args.bilm_mask_last_state = getattr(args, 'bilm_mask_last_state', False)
    args.bilm_add_bos = getattr(args, 'bilm_add_bos', False)



@register_model_architecture('bert_transformer_seq2seq_gan', 'bert_transformer_seq2seq_gan_big')
def bi_transformer_lm_big(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_architecture(args)
