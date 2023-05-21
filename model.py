import abc
import json
import os
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from text.symbols import symbols
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
from torch.cuda.amp import GradScaler

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

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

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
    self.n_position = 1024
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.pre = nn.Conv1d(hidden_channels, hidden_channels, 1)
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    self.n_src_vocab = len(symbols) + 1
    self.src_word_emb = nn.Embedding(
            self.n_src_vocab, hidden_channels
        )
    self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(self.n_position, hidden_channels).unsqueeze(0),
            requires_grad=False,
        )
    self.enc = attentions.FFT(
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout)
    self.norm = nn.LayerNorm(hidden_channels)
    self.act = nn.GELU()

  def forward(self, x, x_lengths):
    batch_size, max_len = x.shape[0], x.shape[1]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)
    # print(x.shape)
    x = self.src_word_emb(x) + self.position_enc[ :, :max_len, : ].expand(batch_size, -1, -1)
    x = x.transpose(1,2)
    # print(x.shape)
    x = self.pre(x)*x_mask
    x = self.norm(x.transpose(1,2)).transpose(1,2)
    # x = self.act(x)*x_mask
    x = self.enc(x * x_mask, x_mask)*x_mask

    x = self.proj(x) * x_mask

    return x
class PromptEncoder(nn.Module):
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
    self.n_position = 1024
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    self.enc = attentions.FFT(
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout)
    self.norm = nn.LayerNorm(hidden_channels)
    self.act = nn.GELU()

  def forward(self, x, x_lengths):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x)*x_mask
    x = self.norm(x.transpose(1,2)).transpose(1,2)
    # x = self.act(x)*x_mask
    x = self.enc(x * x_mask, x_mask)*x_mask
    x = self.proj(x) * x_mask
    return x
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
        self.norm_blocks = nn.ModuleList()
        self.act = nn.ModuleList()
        self.f0_prenet = nn.Conv1d(1, in_channels , 3, padding=1)
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        for _ in range(attention_layers):
            self.conv_blocks.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            self.norm_blocks.append(nn.LayerNorm(hidden_channels))
            self.act.append(nn.GELU())
            self.attn_blocks.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init)
            )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    # MultiHeadAttention 
    def forward(self, x, prompt, x_lenghts, prompt_lenghts):
        x = torch.detach(x)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lenghts, x.size(2)), 1).to(x.dtype)
        prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lenghts, prompt.size(2)), 1).to(prompt.dtype)
        x = self.pre(x) * x_mask
        cross_mask = einsum('b i j, b i k -> b j k', x_mask, prompt_mask).unsqueeze(1)
        # print(x.shape,prompt.shape)
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x) * x_mask
            x = self.norm_blocks[i](x.transpose(1,2)).transpose(1,2) * x_mask
            # x = self.act[i](x) * x_mask
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
        self.norm_blocks = nn.ModuleList()
        self.act = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.f0_prenet = nn.Conv1d(1, in_channels , 3, padding=1)
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        for _ in range(attention_layers):
            self.conv_blocks.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            self.norm_blocks.append(nn.LayerNorm(hidden_channels))
            self.act.append(nn.GELU())
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
            x = self.norm_blocks[i](x.transpose(1,2)).transpose(1,2) * x_mask
            # x = self.act[i](x) * x_mask
            x = x + self.attn_blocks[i](x, prompt, cross_mask) * x_mask
        x = self.proj(x) * x_mask
        return x.squeeze(1)

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
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
      scale=16
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
    self.norm = nn.LayerNorm(hidden_channels)
    self.wn = modules.WN(hidden_channels, kernel_size,
                    dilation_rate, n_layers, gin_channels=self.gin_channels)
    # time condition

    #t/sigma fourier embedding
    sinu_pos_emb = SinusoidalPosEmb(in_channels*dim_time_mult)
    self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(in_channels*dim_time_mult, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
    self.t_norm = nn.LayerNorm(hidden_channels)
    # self.to_t_emb = GaussianFourierProjection(
    #     embedding_size=in_channels*dim_time_mult//2, scale=scale
    # )
    # self.to_time_cond = nn.Linear(in_channels * dim_time_mult, hidden_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)
    self.prompt_norm = nn.LayerNorm(hidden_channels)

  def forward(self, x, data, t):
    contentvec, prompt, contentvec_lengths, prompt_lengths = data
    b, _, _ = x.shape
    x_mask = torch.unsqueeze(commons.sequence_mask(contentvec_lengths, x.size(2)), 1).to(x.dtype)
    prompt2_lengths = torch.Tensor([32 for _ in range(b)]).to(x.device)
    prompt2_mask = torch.unsqueeze(commons.sequence_mask(prompt2_lengths, 32), 1).to(prompt.dtype)
    t = self.time_mlp(t)
    t = self.t_norm(t)
    t = rearrange(t, 'b d -> b 1 d')
    prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lengths, prompt.size(2)), 1).to(prompt.dtype)
    cross_mask = einsum('b i j, b i k -> b j k', prompt2_mask, prompt_mask).unsqueeze(1)
    prompt = self.pre_attn(self.m.expand(b,*self.m.shape),prompt,attn_mask = cross_mask)
    prompt = self.drop(prompt)
    prompt = self.prompt_norm(prompt.transpose(1,2)).transpose(1,2)
    cross2_mask = einsum('b i j, b i k -> b j k', x_mask, prompt2_mask).unsqueeze(1)
    x = self.pre_conv(x) * x_mask
    x = self.norm(x.transpose(1,2)).transpose(1,2)*x_mask
    x_wn = self.wn(x, x_mask, t=t.transpose(1,2),
        cond=contentvec, prompt=prompt, cross_mask=cross2_mask) * x_mask
    x = self.proj(x_wn) * x_mask
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

class Pre_model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.phoneme_encoder = TextEncoder(**self.cfg['phoneme_encoder'])
        self.f0_predictor = F0Predictor(**self.cfg['f0_predictor'])
        self.duration_predictor = DurationPredictor(**self.cfg['duration_predictor'])
        self.prompt_encoder = PromptEncoder(**self.cfg['prompt_encoder'])
        self.f0_emb = nn.Embedding(256, self.cfg['phoneme_encoder']['hidden_channels'])
        self.norm = nn.LayerNorm(self.cfg['phoneme_encoder']['hidden_channels'])
        self.length_regulator = LengthRegulator()
    def forward(self,data, d_control=1.0):
        c_padded, refer_padded, f0_padded, codes_padded, \
        wav_padded, lengths, refer_lengths, text_lengths, \
        uv_padded, phoneme_padded, duration_padded = data
        audio_prompt = self.prompt_encoder(normalize(refer_padded),refer_lengths)


        content_emb = self.phoneme_encoder(phoneme_padded, text_lengths)
        # print(content.shape, duration_padded, text_lengths)
        # print(sum(duration_padded[0]),sum(duration_padded[1]))
        content, mel_len = self.length_regulator(content_emb.transpose(1,2), duration_padded, max_len=f0_padded.shape[1])
        # print(content.transpose(1,2)[0][0])
        # print(content.shape, f0_padded.shape)
        c_mask = torch.unsqueeze(commons.sequence_mask(lengths, f0_padded.size(1)), 1).to(c_padded.dtype)
        content = (content.transpose(1,2) + self.f0_emb(utils.f0_to_coarse(f0_padded)).transpose(1,2))*c_mask
        content = self.norm(content.transpose(1,2)).transpose(1,2)*c_mask

        log_duration_prediction = self.duration_predictor(content_emb, audio_prompt, text_lengths, refer_lengths)
        log_duration_targets = torch.log(duration_padded.float() + 1)
        assert torch.count_nonzero(log_duration_prediction) == torch.count_nonzero(log_duration_targets), 'duration lengths must be the same'

        lf0 = 2595. * torch.log10(1. + f0_padded.unsqueeze(1) / 700.) / 500
        # norm_lf0 = utils.normalize_f0(lf0, c_mask, uv_padded)
        lf0_pred = self.f0_predictor(content, audio_prompt, lengths, refer_lengths)
        # f0_pred = (700 * (torch.pow(10, lf0_pred * 500 / 2595) - 1)).squeeze(1)

        return content, audio_prompt, lf0, lf0_pred, log_duration_prediction, log_duration_targets
    def infer(self, data, d_control=1.0):
        phoneme_padded, refer_padded, text_lengths, refer_lengths = data
        audio_prompt = self.prompt_encoder(normalize(refer_padded),refer_lengths)

        content_emb = self.phoneme_encoder(phoneme_padded, text_lengths)
        log_duration_prediction = self.duration_predictor(content_emb, audio_prompt, text_lengths, refer_lengths)
        # log_duration_targets = torch.log(duration_padded.float() + 1)
        duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
        lengths_pred = torch.sum(duration_rounded, dim=-1)
        max_len = int(max(lengths_pred).item())
        content_emb = self.phoneme_encoder(phoneme_padded, text_lengths)
        content, mel_len = self.length_regulator(content_emb.transpose(1,2), duration_rounded, max_len=max_len)

        lf0_pred = self.f0_predictor(content.transpose(1,2), audio_prompt, lengths_pred, refer_lengths)
        f0_pred = (700 * (torch.pow(10, lf0_pred * 500 / 2595) - 1)).squeeze(1)

        c_mask = torch.unsqueeze(commons.sequence_mask(lengths_pred, max_len), 1).to(content.dtype)
        content = (content.transpose(1,2) + self.f0_emb(utils.f0_to_coarse(f0_pred)).transpose(1,2))*c_mask
        
        return content, audio_prompt, lengths_pred

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

class ReverseDiffusionPredictor():
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean

class LangevinCorrector():
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


def normalize(code):
    return code/10.0
def denormalize(code):
    return code*10.0
class NaturalSpeech2_VESDE(nn.Module):
    def __init__(self,
        cfg,
        rvq_cross_entropy_loss_weight = 1.0,
        diff_loss_weight = 1.0,
        f0_loss_weight = 1.0,
        duration_loss_weight = 1.0,
        scale = 1.,
        ):
        super().__init__()
        self.pre_model = Pre_model(cfg)
        self.diff_model = Diffusion_Encoder(**cfg['diffusion_encoder'])
        self.dim = self.diff_model.in_channels
        self.sampling_timesteps = cfg['train']['sampling_timesteps']

        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.f0_loss_weight = f0_loss_weight
        self.duration_loss_weight = duration_loss_weight

        self.sigma_min = 0.01
        self.sigma_max = 50.0
        self.N = 1000
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.N))
        self.eps=1e-5
        self.T = 1
        self.probability_flow = False
    @property
    def device(self):
        return next(self.diff_model.parameters()).device
    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std
    def forward(self, data, codec):
        c_padded, refer_padded, f0_padded, codes_padded, \
        wav_padded, lengths, refer_lengths, text_lengths, \
        uv_padded, phoneme_padded, duration_padded = data
        batch, d, n, device = *c_padded.shape, self.device
        codes_padded = normalize(codes_padded)
        # get pre model outputs
        content, refer, lf0, lf0_pred,\
        log_duration_prediction, log_duration_targets = self.pre_model(data)
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, codes_padded.size(2)), 1).to(codes_padded.dtype)

        reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        #begin score sde
        t = torch.rand(codes_padded.shape[0], device=codes_padded.device) * (self.T - self.eps) + self.eps
        z = torch.randn_like(codes_padded)*x_mask
        mean, std = self.marginal_prob(codes_padded, t)
        perturbed_data = mean + std[:, None, None]*z

        #get sigmas as labels
        labels = self.marginal_prob(torch.zeros_like(codes_padded), t)[1]
        score = self.diff_model(perturbed_data,(content,refer,lengths,refer_lengths),labels)

        loss_diff = torch.square(score * std[:, None, None] + z)
        loss_diff = reduce_op(loss_diff.reshape(loss_diff.shape[0], -1), dim=-1)
        loss_diff = loss_diff.mean()
        loss_dur = F.mse_loss(log_duration_prediction, log_duration_targets)
        loss_f0 = F.mse_loss(lf0_pred, lf0)

        loss = loss_diff*self.diff_loss_weight + loss_f0*self.f0_loss_weight + loss_dur*self.duration_loss_weight

        # cross entropy loss to codebooks
        ce_loss = torch.tensor(0).float().to(device)
        pred = torch.tensor(0).float().to(device)
        target = torch.tensor(0).float().to(device)

        # _, indices, _, quantized_list = encode(codes_padded,8,codec)
        # ce_loss = rvq_ce_loss(pred.unsqueeze(0)-quantized_list, indices, codec)
        # loss = loss + self.rvq_cross_entropy_loss_weight * ce_loss

        return loss, loss_diff, loss_f0, loss_dur, ce_loss, lf0, lf0_pred, log_duration_prediction, log_duration_targets, pred, target
    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max
    def forward_discretize(self,x,t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                    self.discrete_sigmas.to(t.device)[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G
    def reverse_discretize(self,x,t,data):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = self.forward_discretize(x, t)
        rev_f = f - G[:, None, None] ** 2 * \
            self.diff_model(x,data,t)\
            * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G
    def forward_sde(self,x,t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion
    def reverse_sde(self, x, t, data):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = self.forward_sde(x, t)
        score = self.diff_model(x,data,t)
        drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion
    @torch.no_grad()
    def pc_sample(self, text, refer, text_lengths, refer_lengths, eps=1e-5):
        device = text.device
        sample_steps = self.N//10
        timesteps = torch.linspace(self.T, eps, sample_steps, device=device)
        data = (text, refer, text_lengths, refer_lengths)
        content, refer, lengths = self.pre_model.infer(data)
        shape = (text.shape[0], self.dim, int(lengths.max().item()))
        x = self.prior_sampling(shape).to(device)
        for i in tqdm(range(sample_steps)):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t
            #corrector
            n_steps=1
            target_snr=0.16
            alpha = torch.ones_like(vec_t)
            for i in range(n_steps):
                grad = self.diff_model(x,(content,refer,lengths,refer_lengths),vec_t)
                noise = torch.randn_like(x)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise
            #predictor
            f, G = self.reverse_discretize(x, vec_t,(content,refer,lengths,refer_lengths))
            z = torch.randn_like(x)
            x_mean = x - f
            x = x_mean + G[:, None, None] * z

        return x_mean
    
    @torch.no_grad()
    def sample(self,
        text,
        refer,
        text_lengths,
        refer_lengths,
        codec,
        batch_size = 1):
        sample_fn = self.pc_sample
        audio = sample_fn(text,refer,text_lengths,refer_lengths)

        # print(c.shape, refer.shape, audio.shape)
        audio = audio.transpose(1,2)*8

        audio = codec.decode(audio)

        if audio.ndim == 3:
            audio = rearrange(audio, 'b 1 n -> b n')

        return audio
def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 1.)
# converting gamma to alpha, sigma or logsnr

def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
class NaturalSpeech2_DDPM(nn.Module):
    def __init__(self,
        cfg,
        rvq_cross_entropy_loss_weight = 1.0,
        diff_loss_weight = 1.0,
        f0_loss_weight = 1.0,
        duration_loss_weight = 1.0,
        scale = 1.,
        ddim_sampling_eta = 0.,
        ):
        super().__init__()
        self.pre_model = Pre_model(cfg)
        self.diff_model = Diffusion_Encoder(**cfg['diffusion_encoder'])
        self.dim = self.diff_model.in_channels
        self.sampling_timesteps = cfg['train']['sampling_timesteps']
        self.timesteps = cfg['train']['timesteps']
        self.ddim_sampling_eta = ddim_sampling_eta
        beta_schedule_fn = sigmoid_beta_schedule
        betas = beta_schedule_fn(self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.scale = scale
        self.time_difference = 0.

        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.f0_loss_weight = f0_loss_weight
        self.duration_loss_weight = duration_loss_weight

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, data, clip_x_start = False):
        model_output = self.diff_model(x, data, t)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, text, refer, text_lengths, refer_lengths, return_all_timesteps = False):

        data = (text, refer, text_lengths, refer_lengths)
        content, refer, lengths = self.pre_model.infer(data)
        shape = (text.shape[0], self.dim, int(lengths.max().item()))
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        audio = torch.randn(shape, device = device)
        audios = [audio]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(audio, 
                time_cond,(content,refer,lengths,refer_lengths),
                clip_x_start = True)

            if time_next < 0:
                audio = x_start
                audios.append(audio)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(audio)

            audio = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            audios.append(audio)

        ret = audio if not return_all_timesteps else torch.stack(audios, dim = 1)

        return ret

    @torch.no_grad()
    def sample(self,text, refer, text_lengths, refer_lengths, codec, return_all_timesteps = False):

        sample_fn = self.ddim_sample
        audio = sample_fn(text,refer,text_lengths,refer_lengths)

        audio = audio.transpose(1,2)
        audio = denormalize(audio)
        audio = codec.decode(audio)

        if audio.ndim == 3:
            audio = rearrange(audio, 'b 1 n -> b n')

        return audio 
        
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        return F.l1_loss

    def forward(self, data, codec):
        c_padded, refer_padded, f0_padded, codes_padded, \
        wav_padded, lengths, refer_lengths, text_lengths, \
        uv_padded, phoneme_padded, duration_padded = data
        batch, d, n, device = *c_padded.shape, self.device
        t = torch.randint(0, self.timesteps, (batch,), device=device).long()
        codes_padded = normalize(codes_padded)

        # get pre model outputs
        content, refer, lf0, lf0_pred,\
        log_duration_prediction, log_duration_targets = self.pre_model(data)
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, codes_padded.size(2)), 1).to(codes_padded.dtype)
        x_start = codes_padded
        noise = torch.randn_like(x_start)*x_mask

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        pred = self.diff_model(x,(content,refer,lengths,refer_lengths),t)

        target = x_start

        loss_diff = self.loss_fn(pred, target, reduction = 'none')
        loss_diff = reduce(loss_diff, 'b ... -> b (...)', 'mean').mean()
        loss_dur = F.mse_loss(log_duration_prediction, log_duration_targets)
        loss_f0 = F.mse_loss(lf0_pred, lf0)

        loss = loss_diff*self.diff_loss_weight + loss_f0*self.f0_loss_weight + loss_dur*self.duration_loss_weight

        # cross entropy loss to codebooks
        _, indices, _, quantized_list = encode(denormalize(codes_padded),8,codec)
        ce_loss = rvq_ce_loss(denormalize(pred.unsqueeze(0))-quantized_list, indices, codec)
        loss = loss + self.rvq_cross_entropy_loss_weight * ce_loss

        return loss, loss_diff, loss_f0, loss_dur, ce_loss, lf0, lf0_pred, log_duration_prediction, log_duration_targets, pred, target
    @property
    def device(self):
        return next(self.diff_model.parameters()).device

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
def save_audio(audio, path, codec):
    audio = denormalize(audio)
    audio = audio.unsqueeze(0).transpose(1,2)
    audio = codec.decode(audio)
    if audio.ndim == 3:
        audio = rearrange(audio, 'b 1 n -> b n')
    audio = audio.detach().cpu()

    torchaudio.save(path, audio, 24000)
    
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
class Trainer(object):
    def __init__(
        self,
        cfg_path = './config.json',
        device = 'cuda:1'
    ):
        super().__init__()

        self.cfg = json.load(open(cfg_path))

        self.device = device

        # model
        self.codec = EncodecWrapper().to(device)
        self.codec.eval()
        # self.model = NaturalSpeech2_VESDE(cfg=self.cfg).to(device)
        self.model = NaturalSpeech2_DDPM(cfg=self.cfg).to(device)
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

        self.dl = cycle(dl)
        self.eval_dl = DataLoader(ds, batch_size = 1, shuffle = False, pin_memory = True, num_workers = self.cfg['train']['num_workers'], collate_fn = collate_fn)
        # print(1)
        # optimizer

        self.opt = Adam(self.model.parameters(), lr = self.cfg['train']['train_lr'], betas = self.cfg['train']['adam_betas'])

        # for logging results in a folder periodically

        self.logs_folder = Path(self.cfg['train']['logs_folder'])
        self.logs_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        # self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.scaler = GradScaler()

    def save(self, milestone):
        # if not self.accelerator.is_local_main_process:
        #     return

        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict()
            # 'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.logs_folder / f'model-{milestone}.pt'), map_location=device)

        # model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.scaler.load_state_dict(data['scaler'])

        # if exists(self.accelerator.scaler) and exists(data['scaler']):
        #     self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        # print(1)
        # accelerator = self.accelerator
        device = self.device

        # if accelerator.is_main_process:
        logger = utils.get_logger(self.cfg['train']['logs_folder'])
        writer = SummaryWriter(log_dir=self.cfg['train']['logs_folder'])
        writer_eval = SummaryWriter(log_dir=os.path.join(self.cfg['train']['logs_folder'], "eval"))

        # with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                self.opt.zero_grad()
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data = [d.to(device) for d in data]

                    with torch.autocast(dtype=torch.bfloat16,device_type='cuda'):
                        loss, loss_diff, loss_f0, loss_dur, ce_loss,\
                        lf0, lf0_pred, log_duration_prediction, log_duration_targets,\
                        pred, target = self.model(data, self.codec)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.scaler.scale(loss).backward()
                log_duration_prediction = log_duration_prediction.float()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                pbar.set_description(f'loss: {total_loss:.4f}')

                self.scaler.step(self.opt)
                self.scaler.update()

############################logging#############################################
                # if accelerator.is_main_process and self.step % 100 == 0:
                if self.step%100 == 0:
                    logger.info('Train Epoch: {} [{:.0f}%]'.format(
                        self.step//len(self.ds),
                        100. * self.step / self.train_num_steps))
                    logger.info(f"Losses: {[loss_diff, loss_f0, loss_dur, ce_loss]}, step: {self.step}")

                    scalar_dict = {"loss/diff": loss_diff, "loss/all": total_loss,
                                "loss/f0": loss_f0,"loss/dur":loss_dur, "loss/ce": ce_loss}
                    image_dict = {
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                            lf0_pred[0, 0, :].detach().cpu().numpy()),
                        "all/dur": utils.plot_data_to_numpy(log_duration_targets[0, :].cpu().numpy(),
                                                            log_duration_prediction[0, :].detach().cpu().numpy()),
                    }

                    utils.summarize(
                        writer=writer,
                        global_step=self.step,
                        images=image_dict,
                        scalars=scalar_dict
                    )
                self.step += 1
                # if accelerator.is_main_process:
                if True:
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:

                        # save_audio(pred[1], str(self.logs_folder / f'pred-{self.step}.wav'), self.codec)
                        # save_audio(target[1], str(self.logs_folder / f'target-{self.step}.wav'), self.codec)

                        self.model.eval()
                        c_padded, refer_padded, f0_padded, codes_padded, \
                        wav_padded, lengths, refer_lengths, text_lengths, \
                        uv_padded, phoneme_padded, duration_padded = next(iter(self.eval_dl))
                        text, refer, text_lengths, refer_lengths = phoneme_padded.to(device), normalize(refer_padded).to(device), text_lengths.to(device), refer_lengths.to(device)
                        lengths, refer_lengths = lengths.to(device), refer_lengths.to(device)
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)

                            samples = self.model.sample(text, refer, text_lengths, refer_lengths, self.codec).detach().cpu()

                        # print(samples.shape)
                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), samples, 24000)
                        audio_dict = {}
                        audio_dict.update({
                                f"gen/audio": samples,
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
                        self.model.train()

                pbar.update(1)

        print('training complete')

