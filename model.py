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
    self.enc = attentions.FFT(
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout)

  def forward(self, x, x_lengths, f0=None, noice_scale=1):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x)*x_mask
    if f0 is not None:
        f0 = self.f0_emb(f0).transpose(1,2)*x_mask
        x = (x + f0)*x_mask
    x = self.enc(x * x_mask, x_mask)
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
        self.norm_blocks = nn.ModuleList()
        self.f0_prenet = nn.Conv1d(1, in_channels , 3, padding=1)
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        for _ in range(attention_layers):
            self.conv_blocks.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            self.norm_blocks.append(nn.LayerNorm(hidden_channels))
            self.attn_blocks.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init)
            )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    # MultiHeadAttention 
    def forward(self, x, prompt, norm_f0, x_lenghts, prompt_lenghts):
        x = torch.detach(x)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lenghts, x.size(2)), 1).to(x.dtype)
        prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lenghts, prompt.size(2)), 1).to(prompt.dtype)
        x = (x + self.f0_prenet(norm_f0)) * x_mask
        x = self.pre(x) * x_mask
        cross_mask = einsum('b i j, b i k -> b j k', x_mask, prompt_mask).unsqueeze(1)
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x) * x_mask
            x = self.norm_blocks[i](x.transpose(1,2)).transpose(1,2) * x_mask
            x = x + self.attn_blocks[i](x, prompt, cross_mask) * x_mask
        x = self.proj(x) * x_mask
        return x

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
    self.norm = nn.LayerNorm(hidden_channels)
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
    # self.act = nn.GELU()
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

  def forward(self, x, data, t):
    contentvec, prompt, contentvec_lengths, prompt_lengths = data
    b, _, _ = x.shape
    x_mask = torch.unsqueeze(commons.sequence_mask(contentvec_lengths, x.size(2)), 1).to(x.dtype)
    prompt2_lengths = torch.Tensor([32 for _ in range(b)]).to(x.device)
    prompt2_mask = torch.unsqueeze(commons.sequence_mask(prompt2_lengths, 32), 1).to(prompt.dtype)

    prompt_mask = torch.unsqueeze(commons.sequence_mask(prompt_lengths, prompt.size(2)), 1).to(prompt.dtype)
    t = self.to_time_cond_pre(t)
    if self.cond_time:
        assert exists(t)
        t = self.to_time_cond(t)
        t = rearrange(t, 'b d -> b 1 d')
    cross_mask = einsum('b i j, b i k -> b j k', prompt2_mask, prompt_mask).unsqueeze(1)
    prompt = self.pre_attn(self.m.expand(b,*self.m.shape),prompt, attn_mask=cross_mask)
    prompt = self.drop(prompt)
    cross2_mask = einsum('b i j, b i k -> b j k', x_mask, prompt2_mask).unsqueeze(1)
    x = self.pre_conv(x) * x_mask
    x = self.norm(x.transpose(1,2)).transpose(1,2) * x_mask
    x = self.wn(x, x_mask, t=t.transpose(1,2),
        cond=contentvec, prompt=prompt, cross_mask=cross2_mask) * x_mask
    x = self.proj(x) * x_mask
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


def normalize(code):
    return code/10.0
def denormalize(code):
    return code*10.0
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
        audio_prompt = self.prompt_encoder(normalize(refer_padded),refer_lengths)

        lf0 = 2595. * torch.log10(1. + f0_padded.unsqueeze(1) / 700.) / 500
        norm_lf0 = utils.normalize_f0(lf0, c_mask, uv_padded)
        lf0_pred = self.f0_predictor(c_padded, audio_prompt, norm_lf0, lengths, refer_lengths)
        # f0_pred = (700 * (torch.pow(10, lf0_pred * 500 / 2595) - 1)).squeeze(1)

        content = self.phoneme_encoder(c_padded, lengths,utils.f0_to_coarse(f0_padded))
        
        return content, audio_prompt, lf0, lf0_pred
    def infer(self, data,auto_predict_f0=None):
        c_padded, refer_padded, f0_padded, codes_padded, wav_padded, lengths, refer_lengths, uv_padded = data
        c_mask = torch.unsqueeze(commons.sequence_mask(lengths, c_padded.size(2)), 1).to(c_padded.dtype)
        audio_prompt = self.prompt_encoder(normalize(refer_padded),refer_lengths)

        lf0 = 2595. * torch.log10(1. + f0_padded.unsqueeze(1) / 700.) / 500
        norm_lf0 = utils.normalize_f0(lf0, c_mask, uv_padded)
        lf0_pred = self.f0_predictor(c_padded, audio_prompt, norm_lf0, lengths, refer_lengths)
        f0_pred = (700 * (torch.pow(10, lf0_pred * 500 / 2595) - 1)).squeeze(1)
        if auto_predict_f0 == False:
            f0_pred = f0_padded
        content = self.phoneme_encoder(c_padded, lengths,utils.f0_to_coarse(f0_pred))
        
        return content, audio_prompt

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

def normalize(code):
    return code/10.0
def denormalize(code):
    return code*10.0

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
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
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
class NaturalSpeech2(nn.Module):
    def __init__(self,
        cfg,
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        rvq_cross_entropy_loss_weight = 1.,
        diff_loss_weight = 1.,
        f0_loss_weight = 1.,
        ddim_sampling_eta = 0.,
        scale = 1.,
        ):
        super().__init__()
        self.pre_model = Pre_model(cfg)
        self.diff_model = Diffusion_Encoder(**cfg['diffusion_encoder'])
        self.dim = self.diff_model.in_channels
        self.sampling_timesteps = cfg['train']['sampling_timesteps']
        self.timesteps = cfg['train']['timesteps']
        # gamma schedules
        self.gamma_schedule = sigmoid_schedule
        self.gamma_schedule = partial(self.gamma_schedule)

        self.min_snr_gamma = min_snr_gamma
        self.min_snr_loss_weight = min_snr_loss_weight

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

        # probability for self conditioning during training

        # self.train_prob_self_cond = train_prob_self_cond

        self.scale = scale

        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.f0_loss_weight = f0_loss_weight
    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times
    @property
    def device(self):
        return next(self.diff_model.parameters()).device
    def forward(self, data, codec):
        c_padded, refer_padded, f0_padded, codes_padded, wav_padded, lengths, refer_lengths, uv_padded = data
        codes_padded = normalize(codes_padded)
        batch, d, n, device = *c_padded.shape, self.device
        # predict and take gradient step
        content, refer, lf0, lf0_pred = self.pre_model(data)
        # sample random times
        t = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, codes_padded.size(2)), 1).to(codes_padded.dtype)
        x_start = codes_padded
        noise = torch.randn_like(x_start)*x_mask
        # noise sample
        gamma = self.gamma_schedule(t)
        padded_gamma = right_pad_dims_to(x_start, gamma)
        alpha, sigma =  gamma_to_alpha_sigma(padded_gamma, self.scale)

        x = alpha * x_start + sigma * noise

        pred = self.diff_model(x, (content, refer, lengths, refer_lengths), t)

        target = x_start

        loss_diff = F.mse_loss(pred, target, reduction = 'none')
        loss_diff = reduce(loss_diff, 'b ... -> b', 'mean')
         # min snr loss weight

        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = self.min_snr_gamma)

        loss_weight = maybe_clipped_snr

        loss_diff =  (loss_diff * loss_weight).mean()

        loss_f0 = F.mse_loss(lf0_pred, lf0)

        loss = loss_diff*self.diff_loss_weight + loss_f0*self.f0_loss_weight

        # cross entropy loss to codebooks
        _, indices, _, quantized_list = encode(denormalize(codes_padded),8,codec)
        ce_loss = rvq_ce_loss(denormalize(pred.unsqueeze(0))-quantized_list, indices, codec)
        loss = loss + self.rvq_cross_entropy_loss_weight * ce_loss

        return loss, loss_diff, loss_f0, ce_loss, lf0, lf0_pred
    @torch.no_grad()
    def ddim_sample(self, content, refer, f0, uv, lengths, refer_lengths, shape, time_difference=None, auto_predict_f0=None):
        batch, device = shape[0], self.device

        time_difference = self.time_difference
        time_pairs = self.get_sampling_timesteps(batch, device = device)

        audio = torch.randn(shape, device = device)
        x_start = None

        data = (content, refer, f0, 0, 0, lengths, refer_lengths, uv)
        content, refer = self.pre_model.infer(data,auto_predict_f0=auto_predict_f0)
        # print(audio.shape, content.shape)
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, audio), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)

            # add the time delay

            time_next = (time_next - time_difference).clamp(min = 0.)

            model_output = self.diff_model(audio, (content, refer, lengths, refer_lengths), time)

            x_start = model_output
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
        auto_predict_f0=None):
        sample_fn = self.ddim_sample
        audio = sample_fn(c,refer,f0,uv,lengths,refer_lengths,(1, self.dim, c.shape[-1]),auto_predict_f0 = auto_predict_f0)

        # print(c.shape, refer.shape, audio.shape)
        audio = audio.transpose(1,2)
        audio = denormalize(audio)
        audio = codec.decode(audio)

        if audio.ndim == 3:
            audio = rearrange(audio, 'b 1 n -> b n')

        return audio
    @property
    def loss_fn(self):
        return F.l1_loss

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
        from accelerate import DistributedDataParallelKwargs

        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.cfg = json.load(open(cfg_path))
        self.accelerator = Accelerator(
            # [ddp_kwargs]
        )
        # print(self.accelerator.device)

        device = self.accelerator.device

        # model
        self.codec = EncodecWrapper().cuda()
        self.codec.eval()
        self.model = NaturalSpeech2(cfg=self.cfg).to(device)
        # print(1)
        # sampling and training hyperparameters

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
                            samples = self.ema.ema_model.sample(c, refer, f0, uv, lengths, refer_lengths, self.codec).detach().cpu()

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

                pbar.update(1)

        accelerator.print('training complete')
