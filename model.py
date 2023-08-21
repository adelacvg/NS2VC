import json
import os
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from unet1d.unet_1d_condition import UNet1DConditionModel
from utils import plot_spectrogram_to_numpy
from vocos import Vocos
from torch import expm1, nn
import torchaudio
from dataset import NS2VCDataset, TextAudioCollate, TestDataset
import modules.commons as commons
from accelerate import Accelerator
from parametrizations import weight_norm
from operations import OPERATIONS_ENCODER, MultiheadAttention, TransformerFFNLayer
from accelerate import DistributedDataParallelKwargs
from ema_pytorch import EMA
import math
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import logging
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import utils

from tqdm.auto import tqdm

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

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
        # self.attn_blocks = nn.ModuleList([])
        # self.attn_blocks.extend([
        #     MultiheadAttention(hidden_channels, 8, dropout=self.dropout, bias=False)
        #     for i in range(self.num_layers)
        # ])
        self.last_ln = last_ln
        self.pre = ConvLayer(in_channels, hidden_channels, 1, p_dropout)
        # self.prompt_proj = ConvLayer(in_channels, hidden_channels, 1, p_dropout)
        self.out_proj = ConvLayer(hidden_channels, out_channels, 1, p_dropout)
        if last_ln:
            self.layer_norm = LayerNorm(out_channels)

    def forward(self, src_tokens, lengths, prompt,prompt_lengths):
        # B x C x T -> T x B x C
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
class F0Predictor(nn.Module):
    def __init__(self,
        in_channels=256,
        hidden_channels=512,
        out_channels=1,
        attention_layers=10,
        n_heads=8,
        p_dropout=0.5,):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.n_heads = n_heads
        self.act = nn.ModuleList()
        self.f0_prenet = ConvLayer(1, hidden_channels , kernel_size=3, dropout=p_dropout)
        self.pre = ConvLayer(in_channels, hidden_channels, kernel_size=5, dropout=p_dropout)
        for _ in range(attention_layers):
            self.conv_blocks.append(nn.ModuleList([
                EncConvLayer(hidden_channels, kernel_size=5, dropout=p_dropout),
                EncConvLayer(hidden_channels, kernel_size=5, dropout=p_dropout),
                EncConvLayer(hidden_channels, kernel_size=5, dropout=p_dropout),
            ]))
            self.norm.append(LayerNorm(hidden_channels))
            self.attn_blocks.append(
                MultiheadAttention(hidden_channels, n_heads, dropout=p_dropout, bias=False)
            )
        self.proj = ConvLayer(hidden_channels, out_channels, kernel_size=5, dropout=p_dropout)
        self.dropout = nn.Dropout(p_dropout)
    # MultiHeadAttention 
    def forward(self, x, prompt, norm_f0, x_lenghts, prompt_lenghts):
        norm_f0 = rearrange(norm_f0, 'b c t -> t b c')
        # x = rearrange(x, 'b c t -> t b c')
        x = x.detach()
        prompt = prompt.detach()
        x_mask = ~commons.sequence_mask(x_lenghts, x.size(0)).to(torch.bool)
        prompt_mask = ~commons.sequence_mask(prompt_lenghts, prompt.size(0)).to(torch.bool)
        x = self.pre(x, x_mask)
        x = x + self.f0_prenet(norm_f0, x_mask)
        x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
        prompt = prompt.masked_fill(prompt_mask.t().unsqueeze(-1), 0)
        cross_mask = ~einsum('b j, b k -> b j k', ~x_mask, ~prompt_mask).view(x.shape[1], 1, x_mask.shape[1], prompt_mask.shape[1]).   \
            expand(-1, self.n_heads, -1, -1).reshape(x.shape[1] * self.n_heads, x_mask.shape[1], prompt_mask.shape[1])
        for i in range(len(self.conv_blocks)):
            for conv in self.conv_blocks[i]:
                x = conv(x, x_mask)
            x = self.norm[i](x)
            residual = self.attn_blocks[i](x, prompt, prompt, key_padding_mask = prompt_mask)[0]
            x = x + residual
        assert torch.isnan(x).any() == False
        x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
        x = self.proj(x, x_mask)
        x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
        x = rearrange(x, 't b c -> b c t')
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
        self.f0_predictor = F0Predictor(**self.cfg['f0_predictor'])
        print("f0 params:", count_parameters(self.f0_predictor))
        self.prompt_encoder = PromptEncoder(**self.cfg['prompt_encoder'])
        print("prompt params:", count_parameters(self.prompt_encoder))
        dim = self.cfg['phoneme_encoder']['out_channels']
        self.f0_emb = nn.Embedding(256, dim)
    def forward(self,data):
        c_padded, refer_padded, f0_padded, spec_padded,\
        wav_padded, lengths, refer_lengths, uv_padded = data
        # c_mask = ~commons.sequence_mask(lengths, c_padded.size(0)).to(torch.bool)
        audio_prompt = self.prompt_encoder(normalize(refer_padded),refer_lengths)
        content = self.phoneme_encoder(c_padded, lengths, audio_prompt, refer_lengths)

        lf0 = 2595. * torch.log10(1. + f0_padded.unsqueeze(1) / 700.) / 500
        norm_lf0 = utils.normalize_f0(lf0, uv_padded)
        lf0_pred = self.f0_predictor(content, audio_prompt, norm_lf0, lengths, refer_lengths)
        # f0_pred = (700 * (torch.pow(10, lf0_pred * 500 / 2595) - 1)).squeeze(1)
        f0 = self.f0_emb(utils.f0_to_coarse(f0_padded)).transpose(0,1)
        content = content + f0

        return content, audio_prompt, lf0, lf0_pred
    def infer(self, data,auto_predict_f0=None):
        c_padded, refer_padded, f0_padded, spec_padded, \
        wav_padded, lengths, refer_lengths, uv_padded = data
        # c_mask = ~commons.sequence_mask(lengths, c_padded.size(0)).to(torch.bool)
        audio_prompt = self.prompt_encoder(normalize(refer_padded),refer_lengths)
        content = self.phoneme_encoder(c_padded, lengths, audio_prompt, refer_lengths)

        lf0 = 2595. * torch.log10(1. + f0_padded.unsqueeze(1) / 700.) / 500
        norm_lf0 = utils.normalize_f0(lf0, uv_padded)
        lf0_pred = self.f0_predictor(content, audio_prompt, norm_lf0, lengths, refer_lengths)
        f0_pred = (700 * (torch.pow(10, lf0_pred * 500 / 2595) - 1)).squeeze(1)
        if auto_predict_f0 == False:
            f0_pred = f0_padded
        f0 = self.f0_emb(utils.f0_to_coarse(f0_pred)).transpose(0,1)
        content = content + f0

        return content, audio_prompt

class Diffusion_Encoder(nn.Module):
  def __init__(self,
      in_channels=128,
      out_channels=128,
      hidden_channels=256,
      kernel_size=5,
      dilation_rate=2,
      n_layers=40,
      n_heads=8,
      p_dropout=0.2,
      dim_time_mult=None,
      ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_heads=n_heads
    self.unet = UNet1DConditionModel(
        in_channels=in_channels+hidden_channels,
        out_channels=out_channels,
        block_out_channels=(200,400,800,800),
        norm_num_groups=8,
        cross_attention_dim=hidden_channels,
        attention_head_dim=n_heads,
    )


  def forward(self, x, data, t):
    assert torch.isnan(x).any() == False
    contentvec, prompt, contentvec_lengths, prompt_lengths = data
    _, b, _ = x.shape
    prompt = rearrange(prompt, 't b c -> b t c')
    contentvec = rearrange(contentvec, 't b c -> b c t')
    x = torch.cat([x, contentvec], dim=1)

    # x_mask = commons.sequence_mask(contentvec_lengths, x.size(2)).to(torch.bool)
    prompt_mask = commons.sequence_mask(prompt_lengths, prompt.size(1)).to(torch.bool)
    x = self.unet(x, t, prompt, encoder_attention_mask=prompt_mask)

    return x.sample


def encode(x, n_q = 8, codec=None):
    quantized_out = torch.zeros_like(x)
    residual = x

    all_losses = []
    all_indices = []
    quantized_list = []
    layers = codec.model.quantizer.vq.layers
    n_q = n_q or len(layers)

    for layer in layers[:n_q]:
        quantized_list.append(quantized_out)
        quantized, indices, loss = layer(residual)
        residual = residual - quantized
        quantized_out = quantized_out + quantized

        all_indices.append(indices)
        all_losses.append(loss)
    quantized_list = torch.stack(quantized_list)
    out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
    return quantized_out, out_indices, out_losses, quantized_list
def rvq_ce_loss(residual_list, indices, codec, n_q=8):
    # codebook = codec.model.quantizer.vq.layers[0].codebook
    layers = codec.model.quantizer.vq.layers
    loss = 0.0
    for i,layer in enumerate(layers[:n_q]):
        residual = residual_list[i].transpose(2,1)
        embed = layer.codebook.t()
        dis = -(
            residual.pow(2).sum(2, keepdim=True)
            - 2 * residual @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        indice = indices[i, :, :]
        dis = rearrange(dis, 'b n m -> (b n) m')
        indice = rearrange(indice, 'b n -> (b n)')
        loss = loss + F.cross_entropy(dis, indice)
    return loss
# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def normalize(code):
    return code
def denormalize(code):
    return code

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
class NaturalSpeech2(nn.Module):
    def __init__(self,
        cfg,
        rvq_cross_entropy_loss_weight = 0.1,
        diff_loss_weight = 1.0,
        f0_loss_weight = 1.0,
        duration_loss_weight = 1.0,
        ddim_sampling_eta = 0,
        min_snr_loss_weight = False,
        min_snr_gamma = 5
        ):
        super().__init__()
        self.pre_model = Pre_model(cfg)
        print("pre params: ", count_parameters(self.pre_model))
        self.diff_model = Diffusion_Encoder(**cfg['diffusion_encoder'])
        print("diff params: ", count_parameters(self.diff_model))
        self.dim = self.diff_model.in_channels
        timesteps = cfg['train']['timesteps']

        beta_schedule_fn = linear_beta_schedule
        betas = beta_schedule_fn(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = timesteps

        self.sampling_timesteps = cfg['train']['sampling_timesteps']
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.f0_loss_weight = f0_loss_weight
        self.duration_loss_weight = duration_loss_weight
        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        register_buffer('loss_weight', maybe_clipped_snr)
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, data = None):
        model_output = self.diff_model(x,data, t)
        x_start = model_output
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, data):
        preds = self.model_predictions(x, t, data)
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, data):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, data=data)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, content, refer, lengths, refer_lengths, f0, uv, auto_predict_f0 = True):
        data = (content, refer, f0, 0, 0, lengths, refer_lengths, uv)
        content, refer = self.pre_model.infer(data)
        shape = (content.shape[1], self.dim, content.shape[0])
        batch, device = shape[0], refer.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t, (content,refer,lengths,refer_lengths))
            imgs.append(img)

        ret = img
        return ret

    @torch.no_grad()
    def ddim_sample(self, content, refer, lengths, refer_lengths, f0, uv, auto_predict_f0 = True):
        data = (content, refer, f0, 0, 0, lengths, refer_lengths, uv)
        content, refer = self.pre_model.infer(data,auto_predict_f0=auto_predict_f0)
        shape = (content.shape[1], self.dim, content.shape[0])
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, (content,refer,lengths,refer_lengths))

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img
        return ret

    @torch.no_grad()
    def sample(self,
        c, refer, f0, uv, lengths, refer_lengths, vocos,
        auto_predict_f0=True, sampling_timesteps=200, sample_method='ddim'
        ):
        self.sampling_timesteps = sampling_timesteps
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if sample_method == 'ddpm':
            sample_fn = self.p_sample_loop
        elif sample_method == 'ddim':
            sample_fn = self.ddim_sample
        audio = sample_fn(c, refer, lengths, refer_lengths, f0, uv, auto_predict_f0)

        audio = denormalize(audio)
        vocos.to(audio.device)
        audio = vocos.decode(audio)

        if audio.ndim == 3:
            audio = rearrange(audio, 'b 1 n -> b n')

        return audio 

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, data, vocos):
        c_padded, refer_padded, f0_padded, spec_padded, \
        wav_padded, lengths, refer_lengths, uv_padded = data
        b, d, n, device = *spec_padded.shape, spec_padded.device
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, spec_padded.size(2)), 1).to(spec_padded.dtype)
        x_start = normalize(spec_padded)*x_mask
        # get pre model outputs
        content, refer, lf0, lf0_pred = self.pre_model(data)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        noise = torch.randn_like(x_start)*x_mask
        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        # predict and take gradient step
        model_out = self.diff_model(x,(content,refer,lengths,refer_lengths), t)
        target = x_start

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss_diff = reduce(loss, 'b ... -> b (...)', 'mean')
        loss_diff = loss_diff * extract(self.loss_weight, t, loss.shape)
        loss_diff = loss_diff.mean()

        loss_f0 = F.l1_loss(lf0_pred, lf0)
        loss = loss_diff + loss_f0

        # cross entropy loss to codebooks
        # _, indices, _, quantized_list = encode(codes_padded,8,codec)
        # ce_loss = rvq_ce_loss(denormalize(model_out.unsqueeze(0))-quantized_list, indices, codec)
        # loss = loss + 0.1 * ce_loss

        return loss, loss_diff, loss_f0, \
        lf0, lf0_pred, model_out, target
def get_grad_norm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2) 
    return total_norm
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
class Trainer(object):
    def __init__(
        self,
        cfg_path = './config.json',
    ):
        super().__init__()

        self.cfg = json.load(open(cfg_path))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        device = self.accelerator.device

        # model
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.model = NaturalSpeech2(cfg=self.cfg).to(device)
        # sampling and training hyperparameters

        self.save_and_sample_every = self.cfg['train']['save_and_sample_every']

        self.batch_size = self.cfg['train']['train_batch_size']
        self.gradient_accumulate_every = self.cfg['train']['gradient_accumulate_every']

        self.train_num_steps = self.cfg['train']['train_num_steps']

        # dataset and dataloader
        collate_fn = TextAudioCollate()
        ds = NS2VCDataset(self.cfg['data']['training_files'], self.cfg, self.vocos)
        self.ds = ds
        dl = DataLoader(ds, batch_size = self.cfg['train']['train_batch_size'], shuffle = True, pin_memory = True, num_workers = self.cfg['train']['num_workers'], collate_fn = collate_fn)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        # print(1)
        # optimizer

        self.opt = AdamW(self.model.parameters(), lr = self.cfg['train']['train_lr'], betas = self.cfg['train']['adam_betas'])

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            eval_ds = TestDataset(self.cfg['data']['val_files'], self.cfg, self.vocos)
            self.eval_dl = DataLoader(eval_ds, batch_size = 1, shuffle = False, num_workers = self.cfg['train']['num_workers'])
            self.eval_dl = iter(cycle(self.eval_dl))
            self.ema = EMA(self.model, beta = self.cfg['train']['ema_decay'], update_every = self.cfg['train']['ema_update_every'])
            self.ema.to(self.device)

        now = datetime.now()
        self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
        self.logs_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(model_path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        # print(1)
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            logger = utils.get_logger(self.logs_folder)
            writer = SummaryWriter(log_dir=self.logs_folder)
            writer_eval = SummaryWriter(log_dir=os.path.join(self.logs_folder, "eval"))

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data = [d.to(device) for d in data]

                    with self.accelerator.autocast():
                        loss, loss_diff, \
                        loss_f0, \
                        lf0, lf0_pred, \
                        pred, target\
                         = self.model(data, self.vocos)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                grad_norm = get_grad_norm(self.model)
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()
                # print(loss_diff, loss_f0, ce_loss)
############################logging#############################################
                if accelerator.is_main_process and self.step % 100 == 0:
                    logger.info('Train Epoch: {} [{:.0f}%]'.format(
                        self.step//len(self.ds),
                        100. * self.step / self.train_num_steps))
                    logger.info(f"Losses: {[loss_diff, loss_f0]}, step: {self.step}")

                    scalar_dict = {"loss/diff": loss_diff, "loss/all": total_loss,
                                "loss/f0": loss_f0, "loss/grad": grad_norm}
                    image_dict = {
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                            lf0_pred[0, 0, :].detach().cpu().numpy()),
                        "all/spec": plot_spectrogram_to_numpy(target[0, :, :].detach().unsqueeze(-1).cpu()),
                        "all/spec_pred": plot_spectrogram_to_numpy(pred[0, :, :].detach().unsqueeze(-1).cpu()),
                    }

                    utils.summarize(
                        writer=writer,
                        global_step=self.step,
                        images=image_dict,
                        scalars=scalar_dict
                    )

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        # print(1)
                        # c_padded, refer_padded, f0_padded, spec_padded, wav_padded, lengths, refer_lengths, uv_padded = next(iter(self.eval_dl))
                        data = next(self.eval_dl)
                        c, f0, spec, audio, uv, \
                        c_refer, f0_refer, spec_refer, audio_refer, uv_refer = data

                        c, refer, f0, uv = c.to(device), spec_refer.to(device), f0.to(device), uv.to(device)
                        lengths, refer_lengths = torch.tensor(c.size(2),dtype=torch.long).to(device).unsqueeze(0),\
                            torch.tensor(refer.size(2),dtype=torch.long).to(device).unsqueeze(0)
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            samples = self.ema.ema_model.sample(c, refer, f0, uv, lengths, refer_lengths, self.vocos).detach().cpu()

                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), samples, 24000)
                        audio_dict = {}
                        audio_dict.update({
                                f"gen/audio": samples,
                                f"gt/audio": audio[0],
                                f"refer/audio": audio_refer[0],
                            })
                        utils.summarize(
                            writer=writer_eval,
                            global_step=self.step,
                            audios=audio_dict,
                            audio_sampling_rate=24000
                        )
                        keep_ckpts = self.cfg['train']['keep_ckpts']
                        if keep_ckpts > 0:
                            utils.clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
