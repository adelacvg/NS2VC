import json
import os
from pathlib import Path
from torch import expm1, nn
import torchaudio
from dataset import NS2VCDataset, TextAudioCollate
import modules.attentions as attentions
import modules.commons as commons
import modules.modules as modules
from modules.attentions import MultiHeadAttention
from accelerate import Accelerator
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
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from encodec_wrapper import  EncodecWrapper
import utils

from tqdm.auto import tqdm

def l2norm(t):
    return F.normalize(t, dim = -1)
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim),requires_grad=True)

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

def cycle(dl):
    while True:
        for data in dl:
            yield data

def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)

def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )
# attention pooling

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head), requires_grad=True)
        self.k_scale = nn.Parameter(torch.ones(dim_head), requires_grad=True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask = None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value = True)
            # print(mask.shape, sim.shape)
            mask = rearrange(mask, 'b 1 j -> b 1 1 j')
            # print(1-mask)
            sim = sim.masked_fill(mask==0, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 32,
        num_latents_mean_pooled =32, # number of latents derived from mean pooled representation of the sequence
        max_seq_len = 512,
        ff_mult = 4
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim),requires_grad=True)

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n = num_latents_mean_pooled)
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))
        self.cond_dim = num_latents+num_latents_mean_pooled

    def forward(self, x, mask = None):
        x = x.transpose(1,2)
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device = device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(x, dim = 1, mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim = -2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask = mask) + latents
            latents = ff(latents) + latents

        latents = latents.transpose(1,2)
        return latents

class TextEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      n_layers,
      gin_channels=0,
      filter_channels=None,
      n_heads=None,
      p_dropout=None,
      cond=False):
    super().__init__()
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    if cond==True:
        self.f0_emb = nn.Embedding(256, hidden_channels)

    self.enc_ =  attentions.Encoder(
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout)

  def forward(self, x, x_lengths, f0=None, noice_scale=1):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    # print(x.shape)
    x = self.pre(x)
    if f0 is not None:
        # print(f0.shape)
        # print(x.shape, self.f0_emb(f0).transpose(1,2).shape)
        # x = x + self.f0_emb(f0).transpose(1,2)
        f0 = self.f0_emb(f0).transpose(1,2)
        x = x + f0
    # print(x.shape, x_mask.shape)
    x = self.enc_(x * x_mask, x_mask)
    x = self.proj(x) * x_mask

    return x, f0

# sinusoidal positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim),requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class FiLM(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.gamma = nn.Linear(dim, dim_out)
        self.beta = nn.Linear(dim, dim_out)
    def forward(self, x, cond):
        gamma = self.gamma(cond.transpose(1,2)).transpose(1,2)
        beta = self.beta(cond.transpose(1,2)).transpose(1,2)
        return x * gamma + beta

class F0Predictor(nn.Module):
    def __init__(self,
        in_channels=512,
        hidden_channels=512,
        out_channels=1,
        conv1d_layers=3,
        attention_layers=10,
        n_heads=8,
        p_dropout=0.5,
        proximal_bias = False,
        proximal_init = True):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.f0_prenet = nn.Conv1d(1, in_channels , 3, padding=1)
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        for _ in range(attention_layers):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    modules.LayerNorm(hidden_channels),
                    nn.GELU()
                )
            )
            self.attn_blocks.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init)
            )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    # MultiHeadAttention 
    def forward(self, x, prompt, x_lenghts, prompt_lenghts):
        x = torch.detach(x)
        # x += self.f0_prenet(norm_f0)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lenghts, x.size(2)), 1).to(x.dtype)
        prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lenghts, prompt.size(2)), 1).to(prompt.dtype)
        # print(x_mask)
        # print(x.shape,x_mask.shape)
        x = self.pre(x) * x_mask
        # print(x_mask.shape,prompt_mask.shape)
        cross_mask = einsum('b i j, b i k -> b j k', x_mask, prompt_mask).unsqueeze(1)
        # print(x.shape,prompt.shape)
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x) * x_mask
            x = x + self.attn_blocks[i](x, prompt, cross_mask) * x_mask
        x = self.proj(x) * x_mask
        return x

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len=None):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len=None):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

class DurationPredictor(nn.Module):
    def __init__(self,
        in_channels=256,
        hidden_channels=512,
        out_channels=1,
        conv1d_layers=3,
        attention_layers=10,
        n_heads=8,
        p_dropout=0.5,
        proximal_bias = False,
        proximal_init = True):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.f0_prenet = nn.Conv1d(1, in_channels , 3, padding=1)
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        for _ in range(attention_layers):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    modules.LayerNorm(hidden_channels),
                    nn.GELU()
                )
            )
            self.attn_blocks.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init)
            )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    # MultiHeadAttention 
    def forward(self, x, prompt, x_lenghts, prompt_lenghts):
        x = x.detach()
        # print(x.shape,prompt.shape)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lenghts, x.size(2)), 1).to(x.dtype)
        prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lenghts, prompt.size(2)), 1).to(prompt.dtype)
        x = self.pre(x) * x_mask
        cross_mask = einsum('b i j, b i k -> b j k', x_mask, prompt_mask).unsqueeze(1)
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x) * x_mask
            x = x + self.attn_blocks[i](x, prompt, cross_mask) * x_mask
        x = self.proj(x) * x_mask
        return x.squeeze(1)

class Diffusion_Encoder(nn.Module):
  def __init__(self,
      in_channels,
      cond_channels,
      out_channels,
      hidden_channels,
      kernel_size=5,
      dilation_rate=1,
      n_layers=40,
      n_heads=8,
      proximal_bias = False,
      proximal_init = True,
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
    self.gin_channels = hidden_channels
    self.pre_conv = nn.Conv1d(in_channels, hidden_channels, 1)
    self.pre_attn = MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init)
    self.layers = nn.ModuleList([])
    self.m = nn.Parameter(torch.randn(hidden_channels,32), requires_grad=True)
    self.norm = modules.LayerNorm(hidden_channels)
    self.wn = modules.WN(hidden_channels, kernel_size,
                    dilation_rate, n_layers, gin_channels=self.gin_channels)
    # time condition

    dim_time = in_channels * dim_time_mult

    self.to_time_cond_pre = nn.Sequential(
        LearnedSinusoidalPosEmb(in_channels),
        nn.Linear(in_channels + 1, dim_time),
        nn.SiLU()
    )

    cond_time = exists(dim_time_mult)

    self.to_time_cond = None
    self.cond_time = cond_time

    if cond_time:
        self.to_time_cond = nn.Linear(in_channels * dim_time_mult, hidden_channels)
    self.act = nn.GELU()
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

  def forward(self, x, contentvec, prompt, contentvec_lengths, prompt_lengths, t):
    b, _, _ = x.shape
    # print(x.shape)
    x_mask = torch.unsqueeze(commons.sequence_mask(contentvec_lengths, x.size(2)), 1).to(x.dtype)
    # print(prompt_lengths)
    prompt2_lengths = torch.Tensor([32 for _ in range(b)]).to(x.device)
    prompt2_mask = torch.unsqueeze(commons.sequence_mask(prompt2_lengths, 32), 1).to(prompt.dtype)
    # print(x.shape, contentvec.shape)
    t = self.to_time_cond_pre(t)
    if self.cond_time:
        assert exists(t)
        t = self.to_time_cond(t)
        t = rearrange(t, 'b d -> b 1 d')
    prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lengths, prompt.size(2)), 1).to(prompt.dtype)
    cross_mask = einsum('b i j, b i k -> b j k', prompt2_mask, prompt_mask).unsqueeze(1)
    prompt = self.pre_attn(self.m.expand(b,*self.m.shape),prompt,attn_mask = cross_mask)
    cross2_mask = einsum('b i j, b i k -> b j k', x_mask, prompt2_mask).unsqueeze(1)
    x = self.pre_conv(x) * x_mask
    x = x
    x = self.norm(x)
    x = self.wn(x, x_mask, t=t.transpose(1,2),
        cond=contentvec, prompt=prompt, cross_mask=cross2_mask) * x_mask
    x = self.proj(self.act(x)) * x_mask
    return x



# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_div(numer, denom):
    return numer / denom.clamp(min = 1e-10)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr

def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)


class Pre_model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.phoneme_encoder = TextEncoder(**self.cfg['phoneme_encoder'])
        self.f0_predictor = F0Predictor(**self.cfg['f0_predictor'])
        self.prompt_encoder = TextEncoder(**self.cfg['prompt_encoder'])
    def forward(self,data):
        c_padded, refer_padded, f0_padded, codes_padded, wav_padded, lengths, refer_lengths, uv_padded = data
        c_mask = torch.unsqueeze(commons.sequence_mask(lengths, c_padded.size(2)), 1).to(c_padded.dtype)
        audio_prompt,_ = self.prompt_encoder(refer_padded,refer_lengths)

        lf0 = 2595. * torch.log10(1. + f0_padded.unsqueeze(1) / 700.) / 500
        norm_lf0 = utils.normalize_f0(lf0, c_mask, uv_padded)
        lf0_pred = self.f0_predictor(c_padded, audio_prompt, norm_lf0, lengths, refer_lengths)
        # f0_pred = (700 * (torch.pow(10, lf0_pred * 500 / 2595) - 1)).squeeze(1)

        content,f0_emb = self.phoneme_encoder(c_padded, lengths,utils.f0_to_coarse(f0_padded))
        
        return content, audio_prompt, lf0, lf0_pred, f0_emb
    def infer(self, data):
        c_padded, refer_padded, f0_padded, codes_padded, wav_padded, lengths, refer_lengths, uv_padded = data
        c_mask = torch.unsqueeze(commons.sequence_mask(lengths, c_padded.size(2)), 1).to(c_padded.dtype)
        audio_prompt,_ = self.prompt_encoder(refer_padded,refer_lengths)

        lf0 = 2595. * torch.log10(1. + f0_padded.unsqueeze(1) / 700.) / 500
        norm_lf0 = utils.normalize_f0(lf0, c_mask, uv_padded)
        lf0_pred = self.f0_predictor(c_padded, audio_prompt, norm_lf0, lengths, refer_lengths)
        f0_pred = (700 * (torch.pow(10, lf0_pred * 500 / 2595) - 1)).squeeze(1)

        # content = self.phoneme_encoder(c_padded, lengths,utils.f0_to_coarse(f0_padded))
        content, f0_emb = self.phoneme_encoder(c_padded, lengths,utils.f0_to_coarse(f0_pred))
        
        return content, audio_prompt, f0_emb

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
    # n_q=1
    for i,layer in enumerate(layers[:n_q]):
        residual = residual_list[i].transpose(2,1)
        # residual = rearrange(residual, 'b m n -> (b m) n')
        # print(residual.shape)
        embed = layer.codebook.t()
        # print(embed.shape)
        dis = -(
            residual.pow(2).sum(2, keepdim=True)
            - 2 * residual @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        # embed_ind = dis.max(dim=-1).indices
        # embed_ind = rearrange(embed_ind, '(b m) -> b m', b=residual.shape[0])
        # dis = -torch.cdist(residual, layer.codebook.unsqueeze(0), p=2.0)
        indice = indices[i, :, :]
        # print(indices.shape)
        # print(indice, embed_ind)
        # print(torch.eq(indice, embed_ind).sum())
        dis = rearrange(dis, 'b n m -> (b n) m')
        indice = rearrange(indice, 'b n -> (b n)')
        loss = loss + F.cross_entropy(dis, indice)
    return loss

class NaturalSpeech2(nn.Module):
    def __init__(self,
        cfg,
        noise_schedule = 'sigmoid',
        objective = 'x0',
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        rvq_cross_entropy_loss_weight = 0.01,
        scale = 1.,
        ):
        super().__init__()
        self.pre_model = Pre_model(cfg)
        self.diff_model = Diffusion_Encoder(**cfg['diffusion_encoder'])
        self.dim = self.diff_model.in_channels
        self.sampling_timesteps = cfg['train']['sampling_timesteps']
        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective
        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = cfg['train']['timesteps']
        self.use_ddim = cfg['train']['use_ddim']
        self.scale = scale

        self.time_difference = time_difference

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight
    @property
    def device(self):
        return next(self.diff_model.parameters()).device
    def forward(self, data, codec):
        c_padded, refer_padded, f0_padded, codes_padded, wav_padded, lengths, refer_lengths, uv_padded = data
        batch, d, n, device = *c_padded.shape, self.device
        # assert d == self.dim, f'codec codebook dimension {d} must match model dimensions {self.dim}'
        # sample random times

        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
        
        noise = torch.randn_like(codes_padded)

        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(codes_padded, gamma)
        alpha, sigma =  gamma_to_alpha_sigma(padded_gamma, self.scale)

        noised_audio = alpha * codes_padded + sigma * noise

        # predict and take gradient step
        content, refer, lf0, lf0_pred, f0_emb = self.pre_model(data)
        # print(noised_audio.shape, content.shape, f0_emb.shape)
        pred = self.diff_model(
                    noised_audio,
                    content, refer, 
                    f0_emb,
                    lengths, refer_lengths,
                    times)

        if self.objective == 'eps':
            target = noise

        elif self.objective == 'x0':
            target = codes_padded

        elif self.objective == 'v':
            target = alpha * noise - sigma * codes_padded

        loss_f0 = F.mse_loss(lf0_pred, lf0)
        loss_diff = F.mse_loss(pred, target)*20

        loss = loss_diff + loss_f0

        # min snr loss weight

        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = self.min_snr_gamma)

        if self.objective == 'eps':
            loss_weight = maybe_clipped_snr / snr

        elif self.objective == 'x0':
            loss_weight = maybe_clipped_snr

        elif self.objective == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        loss =  (loss * loss_weight).mean()

        # cross entropy loss to codebooks
        ce_loss = torch.tensor(0).float().to(device)

        if self.objective == 'x0':
            x_start = pred

        elif self.objective == 'eps':
            x_start = safe_div(codes_padded - sigma * pred, alpha)

        elif self.objective == 'v':
            x_start = alpha * codes_padded - sigma * pred

        _, indices, _, quantized_list = encode(codes_padded,8,codec)
        # _,_,_,residual_list = encodec(x_start,8,codec)
        # print(residual_list_gt[0,0,0,:], residual_list[0,0,0,:])
        ce_loss = rvq_ce_loss(x_start.unsqueeze(0)-quantized_list, indices, codec)
        # ce_loss = rvq_ce_loss(codes_padded.unsqueeze(0)-quantized_list, indices, codec)
        loss = loss + self.rvq_cross_entropy_loss_weight * ce_loss

        return loss, loss_diff, loss_f0, ce_loss, lf0, lf0_pred
    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times
    
    @torch.no_grad()
    def ddim_sample(self, content, refer, f0, uv, lengths, refer_lengths, shape, time_difference=None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        audio = torch.randn(shape, device = device)
        # print(audio.shape)
        x_start = None
        last_latents = None
        data = (content, refer, f0, 0, 0, lengths, refer_lengths, uv)
        content, refer, f0_emb = self.pre_model.infer(data)
        # print(audio.shape, content.shape)
        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # get times and noise levels

            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min = 0.)

            # predict x0

            # model_output = self.model(audio, times) # type: ignore
            model_output = self.diff_model(
                    audio,
                    content, refer, 
                    f0_emb,
                    lengths, refer_lengths,
                    times)

            # print(model_output[0][0], audio[0][0])
            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output

            # get predicted noise

            pred_noise = safe_div(audio - alpha * x_start, sigma)

            # calculate x next

            audio = x_start * alpha_next + pred_noise * sigma_next

        return audio
    
    @torch.no_grad()
    def sample(self,
        c,
        refer,
        f0,
        uv,
        lengths,
        refer_lengths,
        codec,
        batch_size = 1):
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        audio = sample_fn(c,refer,f0,uv,lengths,refer_lengths,(batch_size, self.dim, c.shape[-1]))

        # print(c.shape, refer.shape, audio.shape)
        audio = audio.transpose(1,2)*8

        audio = codec.decode(audio)

        if audio.ndim == 3:
            audio = rearrange(audio, 'b 1 n -> b n')

        return audio
        

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
class Trainer(object):
    def __init__(
        self,
        cfg_path = './config.json',
        split_batches = True,
    ):
        super().__init__()

        # accelerator

        self.cfg = json.load(open(cfg_path))
        self.accelerator = Accelerator(
            # split_batches = split_batches,
            # mixed_precision = 'bf16' if self.cfg['train']['bf16'] else 'no'
        )
        # print(self.accelerator.device)

        self.accelerator.native_amp = self.cfg['train']['amp']
        device = self.accelerator.device

        # model
        self.codec = EncodecWrapper().cuda()
        self.codec.eval()
        self.model = NaturalSpeech2(cfg=self.cfg).to(device)
        # print(1)
        # sampling and training hyperparameters

        assert has_int_squareroot(self.cfg['train']['num_samples']), 'number of samples must have an integer square root'
        self.num_samples = self.cfg['train']['num_samples']
        self.save_and_sample_every = self.cfg['train']['save_and_sample_every']

        self.batch_size = self.cfg['train']['train_batch_size']
        self.gradient_accumulate_every = self.cfg['train']['gradient_accumulate_every']

        self.train_num_steps = self.cfg['train']['train_num_steps']

        # dataset and dataloader
        collate_fn = TextAudioCollate()
        ds = NS2VCDataset(self.cfg, self.codec)
        self.ds = ds
        dl = DataLoader(ds, batch_size = self.cfg['train']['train_batch_size'], shuffle = True, pin_memory = True, num_workers = self.cfg['train']['num_workers'], collate_fn = collate_fn)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.eval_dl = DataLoader(ds, batch_size = 1, shuffle = False, pin_memory = True, num_workers = self.cfg['train']['num_workers'], collate_fn = collate_fn)
        # print(1)
        # optimizer

        self.opt = Adam(self.model.parameters(), lr = self.cfg['train']['train_lr'], betas = self.cfg['train']['adam_betas'])

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = self.cfg['train']['ema_decay'], update_every = self.cfg['train']['ema_update_every'])
            self.ema.to(self.device)

        self.logs_folder = Path(self.cfg['train']['logs_folder'])
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

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.logs_folder / f'model-{milestone}.pt'), map_location=device)

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
            logger = utils.get_logger(self.cfg['train']['logs_folder'])
            writer = SummaryWriter(log_dir=self.cfg['train']['logs_folder'])
            writer_eval = SummaryWriter(log_dir=os.path.join(self.cfg['train']['logs_folder'], "eval"))

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data = [d.to(device) for d in data]

                    with self.accelerator.autocast():
                        loss, loss_diff, loss_f0, ce_loss, lf0, lf0_pred = self.model(data, self.codec)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

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
                    logger.info(f"Losses: {[loss_diff, loss_f0, ce_loss]}, step: {self.step}")

                    scalar_dict = {"loss/diff": loss_diff, "loss/all": total_loss,
                                "loss/f0": loss_f0, "loss/ce": ce_loss}
                    image_dict = {
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                            lf0_pred[0, 0, :].detach().cpu().numpy()),
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

                        c_padded, refer_padded, f0_padded, codes_padded, wav_padded, lengths, refer_lengths, uv_padded = next(iter(self.eval_dl))
                        c, refer, f0, uv = c_padded.to(device), refer_padded.to(device), f0_padded.to(device), uv_padded.to(device)
                        lengths, refer_lengths = lengths.to(device), refer_lengths.to(device)
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(c, refer, f0, uv, lengths, refer_lengths, self.codec, batch_size=n), batches))    

                        all_samples = torch.cat(all_samples_list, dim = 0).detach().cpu()

                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), all_samples, 24000)
                        audio_dict = {}
                        audio_dict.update({
                                f"gen/audio": all_samples,
                                f"gt/audio": wav_padded[0]
                            })
                        utils.summarize(
                            writer=writer_eval,
                            global_step=self.step,
                            audios=audio_dict,
                            audio_sampling_rate=24000
                        )
                        keep_ckpts = self.cfg['train']['keep_ckpts']
                        if keep_ckpts > 0:
                            utils.clean_checkpoints(path_to_models=self.cfg['train']['logs_folder'], n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
