import json
from pathlib import Path
from torch import expm1, nn
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

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from audiolm_pytorch import SoundStream, EncodecWrapper
from audiolm_pytorch.data import SoundDataset, get_dataloader

from beartype import beartype
from beartype.typing import Tuple, Union, Optional

from tqdm.auto import tqdm
# def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
#                sample_rate: int, rescale: bool = False):
#     limit = 0.99
#     mx = wav.abs().max()
#     if rescale:
#         wav = wav * min(limit / mx, 1)
#     else:
#         wav = wav.clamp(-limit, limit)
#     torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
# # Instantiate a pretrained EnCodec model
# model = EncodecModel.encodec_model_24khz()
# # The number of codebooks used will be determined bythe bandwidth selected.
# # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
# model.set_target_bandwidth(24.0)

# # Load and pre-process the audio waveform
# wav, sr = torchaudio.load("1.wav")
# wav = convert_audio(wav, sr, model.sample_rate, model.channels)
# wav = wav.unsqueeze(0)

# # Extract discrete codes from EnCodec
# with torch.no_grad():
#     encoded_frames = model.encode(wav)
# # print(encoded_frames[0].shape)
# codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
# print(codes.shape)
# codes = torch.sum(codes, dim=1)  # [B, T]
# # print(codes.shape)
# wav = model.decode(encoded_frames)
# # print(wav)
# # print(wav.shape)
# save_audio(wav[0].detach(), "1_out.wav", model.sample_rate, rescale=True)
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
        self.gamma = nn.Parameter(torch.ones(dim))

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

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

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
            sim = sim.masked_fill(1-mask, max_neg_value)

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

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

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
        x = x + self.f0_emb(f0).transpose(1,2)
    # print(x.shape, x_mask.shape)
    x = self.enc_(x * x_mask, x_mask)
    x = self.proj(x) * x_mask

    return x

# sinusoidal positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

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
class Diffusion_Encoder(nn.Module):
  def __init__(self,
      in_channels,
      cond_channels,
      out_channels,
      hidden_channels,
      kernel_size=5,
      dilation_rate=1,
      n_layers=10,
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
    self.gin_channels = dim_time_mult*in_channels
    self.pre_conv = nn.Conv1d(in_channels+cond_channels, hidden_channels, 1)
    self.pre_attn = MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init)
    self.layers = nn.ModuleList([])
    self.m = nn.Parameter(torch.randn(512,32))
    for _ in range(n_layers):
        self.layers.append(
            nn.ModuleList([
                modules.LayerNorm(hidden_channels),
                modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=self.gin_channels),
                modules.LayerNorm(hidden_channels),
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init),
                FiLM(hidden_channels, hidden_channels),
            ])
        )
    
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
    self.attn_pool = PerceiverResampler(hidden_channels, depth=4)

    if cond_time:
        self.to_time_cond = nn.Linear(in_channels * dim_time_mult, hidden_channels * 4)

    self.to_text_non_attn_cond = nn.Sequential(
        modules.LayerNorm(hidden_channels),
        nn.Linear(self.attn_pool.cond_dim, 8),
        nn.SiLU(),
        nn.Linear(8, 1)
    )

  def forward(self, x, contentvec, prompt, contentvec_lengths, prompt_lengths, t):
    b, _, _ = x.shape
    x_mask = torch.unsqueeze(commons.sequence_mask(contentvec_lengths, x.size(2)), 1).to(x.dtype)
    prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lengths, prompt.size(2)), 1).to(prompt.dtype)
    x = torch.cat((x, contentvec), dim=1)
    x = self.pre_conv(x) * x_mask
    t = self.to_time_cond_pre(t)
    if self.cond_time:
        assert exists(t)
        t = self.to_time_cond(t)
        t = rearrange(t, 'b d -> b 1 d')
        t_wn_gamma, t_wn_beta, t_attn_gamma, t_attn_beta = t.chunk(4, dim = -1)
    
    cross_mask = einsum('b i j, b i k -> b j k', x_mask, prompt_mask).unsqueeze(1)
    # print(prompt.shape, self.m.shape, self.m.expand(b,*self.m.shape).shape)
    prompt = self.pre_attn(prompt, self.m.expand(b,*self.m.shape)) * prompt_mask
    for norm1, wn, norm2, attn, film in self.layers:
        if self.cond_time:
            # print(x.shape, t_wn_gamma.shape, t_wn_beta.shape)
            x = x * t_wn_gamma.transpose(1,2) + t_wn_beta.transpose(1,2)
        x = norm1(x)
        x = wn(x, x_mask, g=t.transpose(1,2)) * x_mask
        if self.cond_time:
            x = x*t_attn_gamma.transpose(1,2) + t_attn_beta.transpose(1,2)
        x = norm2(x)
        z = attn(x, prompt, attn_mask=cross_mask)
        z = self.attn_pool(z, x_mask)
        z = self.to_text_non_attn_cond(z)
        # print(z.shape, x.shape)
        x = film(x,z) * x_mask
    return z

class F0Predictor(nn.Module):
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
    # MultiHeadAttention
    def forward(self, x, prompt, x_lenghts, prompt_lenghts):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lenghts, x.size(2)), 1).to(x.dtype)
        prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lenghts, prompt.size(2)), 1).to(prompt.dtype)
        # print(x.shape,x_mask.shape)
        x = self.pre(x) * x_mask
        # print(x_mask.shape,prompt_mask.shape)
        cross_mask = einsum('b i j, b i k -> b j k', x_mask, prompt_mask).unsqueeze(1)
        # print(x.shape,prompt.shape)
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x) * x_mask
            x = x + self.attn_blocks[i](x, prompt, cross_mask) * x_mask
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
class NaturalSpeech2(nn.Module):
    def __init__(self,
        codec: Optional[Union[SoundStream, EncodecWrapper]] = None,
        timesteps = 1000,
        use_ddim = True,
        noise_schedule = 'sigmoid',
        objective = 'x0',
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        rvq_cross_entropy_loss_weight = 0.,
        cfg_path = 'config.json',
        is_raw_audio = True,
        train_prob_self_cond = 0.9,
        scale = 1.,
        ):
        super().__init__()
        self.cfg = json.load(cfg_path)
        self.codec = codec
        self.phoneme_encoder = TextEncoder(self.cfg['phoneme_encoder'])
        self.f0_predictor = F0Predictor(self.cfg['f0_predictor'])
        self.prompt_encoder = TextEncoder(self.cfg['prompt_encoder'])
        self.diff_model = Diffusion_Encoder(self.cfg['diffusion_encoder'])
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

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        self.time_difference = time_difference
        self.is_raw_audio = is_raw_audio

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight
    @property
    def device(self):
        return next(self.model.parameters()).device
    def forward(self, audio, audio_mask,
            contentvec, contentvec_length,
            audio_prompt, audio_prompt_length,
            code, code_length,
            f0, uv, codes=None):
        is_raw_audio = self.is_raw_audio
        if is_raw_audio:
            with torch.no_grad():
                self.codec.eval()
                audio, codes, _ = self.codec(audio, return_encoded = True)
        batch, n, d, device = *audio.shape, self.device
        assert d == self.dim, f'codec codebook dimension {d} must match model dimensions {self.dim}'
        # sample random times

        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
        
        noise = torch.randn_like(audio)

        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(audio, gamma)
        alpha, sigma =  gamma_to_alpha_sigma(padded_gamma, self.scale)

        noised_audio = alpha * audio + sigma * noise

        # predict and take gradient step
        audio_prompt = self.prompt_encoder(audio_prompt,audio_prompt_length)

        f0_pred = self.f0_predictor(contentvec, audio_prompt, contentvec_length, audio_prompt_length)

        content = self.phoneme_encoder(
            contentvec, contentvec_length,
            audio_prompt, audio_prompt_length,f0)
        pred = self.diff_model(
            content, contentvec_length,
            audio_prompt, audio_prompt_length,
            noised_audio, times)


        if self.objective == 'eps':
            target = noise

        elif self.objective == 'x0':
            target = audio

        elif self.objective == 'v':
            target = alpha * noise - sigma * audio

        loss_f0 = F.mse_loss(f0_pred, f0)
        loss_diff = F.mse_loss(pred, target)

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

        if self.rvq_cross_entropy_loss_weight == 0 or not exists(codes):
            return loss

        if self.objective == 'x0':
            x_start = pred

        elif self.objective == 'eps':
            x_start = safe_div(audio - sigma * pred, alpha)

        elif self.objective == 'v':
            x_start = alpha * audio - sigma * pred

        _, ce_loss = self.codec.rq(x_start, codes)

        return loss + self.rvq_cross_entropy_loss_weight * ce_loss
    
    @torch.no_grad()
    def ddpm(self, shape, time_difference=None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        audio = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min = 0.)

            noise_cond = time

            # get predicted x0

            model_output = self.model(audio, noise_cond)

            # get log(snr)

            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            # get alpha sigma of time and next time

            alpha, sigma = gamma_to_alpha_sigma(gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next, self.scale)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(audio - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * audio - sigma * model_output

            # derive posterior mean and variance

            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (audio * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(audio),
                torch.zeros_like(audio)
            )

            audio = mean + (0.5 * log_variance).exp() * noise

        return audio
    
    @torch.no_grad()
    def ddim(self, shape, time_difference=None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        audio = torch.randn(shape, device = device)

        x_start = None
        last_latents = None

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

            model_output = self.model(audio, times) # type: ignore

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
        *,
        length,
        batch_size = 1):
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        audio = sample_fn((batch_size, length, self.dim))

        if exists(self.codec):
            audio = self.codec.decode(audio)

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

class Trainer(object):
    def __init__(
        self,
        diffusion_model: NaturalSpeech2,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        self.dim = diffusion_model.dim

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

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
            # 'version': self.__version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_samples = torch.cat(all_samples_list, dim = 0)

                        torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

