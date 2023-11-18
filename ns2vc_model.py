from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from modules.parametrizations import weight_norm
from modules.operations import OPERATIONS_ENCODER, MultiheadAttention, TransformerFFNLayer
from accelerate import DistributedDataParallelKwargs
from torch import nn
import torch
import math
import modules.commons as commons
import torch.nn.functional as F
from modules.perceiver import Attention
from unet1d.embeddings import TextTimeEmbedding
from unet1d.unet_1d_condition import UNet1DConditionModel
from modules.ttts_utils import AttentionBlock, normalization

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, layer, hidden_size, dropout):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.op = OPERATIONS_ENCODER[layer](hidden_size, dropout)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
class ConvTBC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)

class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dropout=0):
        super().__init__()
        self.layer_norm = LayerNorm(c_in)
        conv = ConvTBC(c_in, c_out, kernel_size, padding=kernel_size // 2)
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * c_in))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)
        self.conv = conv

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.t().unsqueeze(-1), 0)
        x = self.layer_norm(x)
        x = self.conv(x)
        return x

class PhoneEncoder(nn.Module):
    def __init__(self,
      in_channels=128,
      hidden_channels=512,
      out_channels=512,
      n_layers=6,
      p_dropout=0.2,
      last_ln = True):
        super().__init__()
        self.arch = [8 for _ in range(n_layers)]
        self.num_layers = n_layers
        self.hidden_size = hidden_channels
        self.padding_idx = 0
        self.dropout = p_dropout
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.last_ln = last_ln
        self.pre = ConvLayer(in_channels, hidden_channels, 1, p_dropout)
        # self.prompt_proj = ConvLayer(in_channels, hidden_channels, 1, p_dropout)
        self.out_proj = ConvLayer(hidden_channels, out_channels, 1, p_dropout)
        if last_ln:
            self.layer_norm = LayerNorm(out_channels)
        self.spk_proj = nn.Conv1d(100,in_channels,1)

    def forward(self, src_tokens, lengths=None, g=None):
        if lengths==None:
            lengths = torch.tensor(src_tokens.shape[-1]).repeat(src_tokens.shape[0]).to(src_tokens.device)
        # B x C x T -> T x B x C
        src_tokens = src_tokens + self.spk_proj(g)
        src_tokens = rearrange(src_tokens, 'b c t -> t b c')
        # compute padding mask
        encoder_padding_mask = ~commons.sequence_mask(lengths, src_tokens.size(0)).to(torch.bool)
        # prompt_mask = ~commons.sequence_mask(prompt_lengths, prompt.size(0)).to(torch.bool)
        x = src_tokens

        x = self.pre(x, encoder_padding_mask=encoder_padding_mask)
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        # prompt = self.prompt_proj(prompt, encoder_padding_mask=prompt_mask)
        # encoder layers
        for i in range(self.num_layers):
            x = self.layers[i](x, encoder_padding_mask=encoder_padding_mask)
            # x = x+self.attn_blocks[i](x, prompt, prompt, key_padding_mask=prompt_mask)[0]
        x = self.out_proj(x, encoder_padding_mask=encoder_padding_mask)
        if self.last_ln:
            x = self.layer_norm(x)
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x

class PromptEncoder(nn.Module):
    def __init__(self,
      in_channels=128,
      hidden_channels=256,
      out_channels=512,
      n_layers=6,
      p_dropout=0.2,
      last_ln = True):
        super().__init__()
        self.arch = [8 for _ in range(n_layers)]
        self.num_layers = n_layers
        self.hidden_size = hidden_channels
        self.padding_idx = 0
        self.dropout = p_dropout
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.last_ln = last_ln
        if last_ln:
            self.layer_norm = LayerNorm(out_channels)
        self.pre = ConvLayer(in_channels, hidden_channels, 1, p_dropout)
        self.out_proj = ConvLayer(hidden_channels, out_channels, 1, p_dropout)

    def forward(self, src_tokens, lengths=None):
        if lengths==None:
            lengths = torch.tensor(src_tokens.shape[-1]).repeat(src_tokens.shape[0]).to(src_tokens.device)
        # B x C x T -> T x B x C
        src_tokens = rearrange(src_tokens, 'b c t -> t b c')
        # compute padding mask
        encoder_padding_mask = ~commons.sequence_mask(lengths, src_tokens.size(0)).to(torch.bool)
        x = src_tokens

        x = self.pre(x, encoder_padding_mask=encoder_padding_mask)
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        x = self.out_proj(x, encoder_padding_mask=encoder_padding_mask)

        if self.last_ln:
            x = self.layer_norm(x)
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x

class EncConvLayer(nn.Module):
    def __init__(self, c, kernel_size, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(c)
        conv = ConvTBC(c, c, kernel_size, padding=kernel_size // 2)
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * c))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)
        self.conv = weight_norm(conv, dim=2)
        self.dropout = dropout
    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        residual = x
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.t().unsqueeze(-1), 0)
        x = self.layer_norm(x)
        x = self.conv(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = x + residual
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)
class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, kernel_size, dropout):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    if dilation==1:
        padding = kernel_size//2
    else:
        padding = dilation
    self.dilated_conv = ConvLayer(residual_channels, 2 * residual_channels, kernel_size)
    self.conditioner_projection = ConvLayer(n_mels, 2 * residual_channels, 1)
    # self.output_projection = ConvLayer(residual_channels, 2 * residual_channels, 1)
    self.output_projection = ConvLayer(residual_channels, residual_channels, 1)
    self.t_proj = ConvLayer(residual_channels, residual_channels, 1)
    self.drop = nn.Dropout(dropout)

  def forward(self, x, diffusion_step, conditioner,x_mask):
    assert (conditioner is None and self.conditioner_projection is None) or \
           (conditioner is not None and self.conditioner_projection is not None)
    #T B C
    y = x + self.t_proj(diffusion_step.unsqueeze(0))
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)
    conditioner = self.conditioner_projection(conditioner)
    conditioner = self.drop(conditioner)
    y = self.dilated_conv(y) + conditioner
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)

    gate, filter_ = torch.chunk(y, 2, dim=-1)
    y = torch.sigmoid(gate) * torch.tanh(filter_)
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)

    y = self.output_projection(y)
    return y
    # y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)
    # residual, skip = torch.chunk(y, 2, dim=-1)
    # return (x + residual) / math.sqrt(2.0), skip

class Pre_model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.phoneme_encoder = PhoneEncoder(**self.cfg['phoneme_encoder'])
        print("phoneme params:", count_parameters(self.phoneme_encoder))
        self.prompt_encoder = PromptEncoder(**self.cfg['prompt_encoder'])
        print("prompt params:", count_parameters(self.prompt_encoder))
        dim = self.cfg['phoneme_encoder']['out_channels']
        self.ref_enc = TextTimeEmbedding(100, 100, 1)
    def forward(self, cvec, prompt):
        g = self.ref_enc(prompt.transpose(1,2)).unsqueeze(-1)
        audio_prompt = self.prompt_encoder(prompt)
        content = self.phoneme_encoder(cvec, g=g)
        content = rearrange(content, 't b c -> b c t')
        audio_prompt = rearrange(audio_prompt, 't b c -> b c t')
        return content, audio_prompt

class Timbre_injector(nn.Module):
  def __init__(self,
      cfg,
      in_channels=128,
      out_channels=128,
      hidden_channels=256,
      n_heads=8,
      p_dropout=0.2,
      ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.n_heads=n_heads
    self.unet = UNet1DConditionModel(
        in_channels=in_channels,
        out_channels=out_channels,
        block_out_channels=(128,128,256,256),
        norm_num_groups=8,
        cross_attention_dim=hidden_channels,
        attention_head_dim=n_heads,
        addition_embed_type='text',
        resnet_time_scale_shift='scale_shift',
    )
    self.pre_model = Pre_model(cfg)


  def forward(self, cvec, prompt):
    _, b, _ = cvec.shape
    cvec, prompt = self.pre_model(cvec, prompt)
    x = self.unet(cvec, 1, prompt.transpose(1,2))

    return x.sample

class CrossAttn(nn.Module):
    def __init__(self,
            hidden_dim,
            dim_context,
            dim_head,
            heads) -> None:
        super().__init__()
        self.attn = Attention(hidden_dim,dim_context=hidden_dim,dim_head=hidden_dim//heads,heads=heads)
        self.norm = normalization(hidden_dim)
    def forward(self, x, context):
        x = self.norm(x)
        x = rearrange(x,'b c t->b t c')
        context = rearrange(context,'b c t->b t c')
        x = x+self.attn(x, context)
        x = rearrange(x,'b t c->b c t')
        return x

class Transformer_blocks(nn.Module):
    def __init__(self, dim, hidden_dim=1024, heads=16, num_blocks=6):
        super().__init__()
        self.num_blocks = num_blocks
        self.x_proj = nn.Linear(dim, hidden_dim)
        self.self_attn_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, heads, relative_pos_embeddings=True) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.x_proj(x.transpose(1,2)).transpose(1,2)
        for i in range(self.num_blocks):
            x = self.self_attn_layers[i](x)
        return x

if __name__ == '__main__':
    model = Transformer_blocks(512,512,1024,1024,16,6)
    output = model(
        torch.rand(4,512,100),
        torch.rand(4,512,100)
    )