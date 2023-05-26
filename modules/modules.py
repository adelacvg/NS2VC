import copy
import math
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm
from modules.attentions import MultiHeadAttention
import modules.commons as commons
from modules.commons import init_weights, get_padding


LRELU_SLOPE = 0.1


class WN(torch.nn.Module):
  def __init__(self, hidden_channels, 
      kernel_size, dilation_rate,
      n_layers, gin_channels=0, p_dropout=0):
    super(WN, self).__init__()
    assert(kernel_size % 2 == 1)
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.attn = torch.nn.ModuleList()
    self.linear = torch.nn.ModuleList()
    self.norm = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    self.cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)

######FiLM
    self.prompt_layer = torch.nn.Conv1d(hidden_channels, hidden_channels*n_layers, 1)

    self.t_layer = torch.nn.Conv1d(hidden_channels, hidden_channels*n_layers, 1)
######FiLM

    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      # in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # norm_layer = nn.LayerNorm(hidden_channels)
      # self.norm.append(norm_layer)
      ####FiLM
      attn = MultiHeadAttention(hidden_channels, hidden_channels, n_heads=8, p_dropout=p_dropout)
      self.attn.append(attn)

      linear = nn.Conv1d(hidden_channels, 2,1)
      self.linear.append(linear)
      ####FiLM
      # last one is not necessary
      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, t=None, 
    cond=None, prompt=None,cross_mask=None, **kwargs):
    output = torch.zeros_like(x)
    n_channels_tensor = torch.IntTensor([self.hidden_channels])

    cond = self.cond_layer(cond)
    prompt = self.prompt_layer(prompt)
    t = self.t_layer(t)

    for i in range(self.n_layers):
      cond_offset = i * self.hidden_channels
      x_t = (x + t[:,cond_offset:cond_offset+self.hidden_channels,:])*x_mask
      # x_t = (x + t)*x_mask
      x_in = self.in_layers[i](x_t)*x_mask


      if cond is not None:
        cond_offset = i * 2 * self.hidden_channels
        cond_l = cond[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        cond_l = torch.zeros_like(x_in)
      # print(x_in.shape,cond_l.shape,content.shape)
      x_in = (x_in + cond_l)*x_mask
      # x_in = (x_in+cond)*x_mask

      ########FiLM########
      cond_offset = i * self.hidden_channels
      scale_shift = self.attn[i](x_t,prompt[:,cond_offset:cond_offset+self.hidden_channels,:],cross_mask)*x_mask
      # scale_shift = self.attn[i](x_t, prompt,cross_mask)*x_mask
      scale_shift = self.linear[i](scale_shift)*x_mask
      scale, shift = scale_shift.chunk(2, dim=1)
      x_in = (x_in * scale + shift)*x_mask
      ########FiLM########
      acts = commons.fused_add_tanh_sigmoid_multiply(x_in, n_channels_tensor)
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)*x_mask
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:,self.hidden_channels:,:]
      else:
        output = output + res_skip_acts
    return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
     torch.nn.utils.remove_weight_norm(l)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Log(nn.Module):
  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
      return x
    
class ElementwiseAffine(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.channels = channels
    self.m = nn.Parameter(torch.zeros(channels,1))
    self.logs = nn.Parameter(torch.zeros(channels,1))

  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = self.m + torch.exp(self.logs) * x
      y = y * x_mask
      logdet = torch.sum(self.logs * x_mask, [1,2])
      return y, logdet
    else:
      x = (x - self.m) * torch.exp(-self.logs) * x_mask
      return x
  