from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    normalization,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, Upsample, convert_module_to_f16, convert_module_to_f32
from ldm.util import exists
import torch as th
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from cldm.cond_emb import CLIP

from aa_utils.aa_utils import normalization, AttentionBlock

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BaseModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        hint_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=1,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        self.blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            # if level != len(channel_mult) - 1:
            out_ch = ch
            self.blocks.append(
                TimestepEmbedSequential(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
            )
            ch = out_ch
            input_block_chans.append(ch)
            # ds *= 2
            self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        self.hint_converter = nn.Conv1d(hint_channels,model_channels,3,padding=1)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)
        # self.input_blocks.apply(convert_module_to_f16)
        # self.middle_block.apply(convert_module_to_f16)
        # self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)
        # self.input_blocks.apply(convert_module_to_f32)
        # self.middle_block.apply(convert_module_to_f32)
        # self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, hint=None, control=None,  **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        # guided_hint = self.input_hint_block(hint, emb, context)
        hint = self.hint_converter(hint)
        # context = self.context_proj(context).unsqueeze(-1)
        # scale, shift = torch.chunk(context, 2, dim = 1)
        # hint = hint*(1+scale)+shift
        h = x.type(self.dtype)
        flag=0
        for module in self.blocks:
            if flag==0:
                h = module(h, emb, context, control.pop(0))
                h += hint
                flag=1
            else:
                h = module(h, emb, context, control.pop(0))
            hs.append(h)
        h = h.type(x.dtype)
        return self.out(h)

class ReferenceNet(BaseModel):
    def forward(self, x, timesteps=None, context=None, **kwargs):
        hs = []
        control = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.blocks:
            h,refer = module(h, emb, context,return_refer=True)
            hs.append(h)
            control.append(refer)
        h = h.type(x.dtype)
        # h = self.out(h)
        return control

TACOTRON_MEL_MAX = 5.5451774444795624753378569716654
TACOTRON_MEL_MIN = -16.118095650958319788125940182791
# TACOTRON_MEL_MIN = -11.512925464970228420089957273422

CVEC_MAX = 5.5451774444795624753378569716654
CVEC_MIN = -5.5451774444795624753378569716654
def denormalize_tacotron_mel(norm_mel):
    return norm_mel/0.18215
def normalize_tacotron_mel(mel):
    mel = torch.clamp(mel, min=-TACOTRON_MEL_MAX)
    return mel*0.18215

def denormalize_cvec(norm_mel):
    return norm_mel/0.11111
def normalize_cvec(mel):
    return mel*0.11111

class AA_diffusion(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refer_enc = CLIP(**config['clip'])
        self.refer_model = ReferenceNet(**config['refer_diffusion'])
        self.base_model = BaseModel(**config['base_diffusion'])
        print("base model params:", count_parameters(self.base_model))
        self.unconditioned_percentage = 0.1
        # self.control_model = instantiate_from_config(control_stage_config)
        # self.refer_model = instantiate_from_config(refer_config)
        self.control_scales = [1.0] * 13
        # self.unconditioned_embedding = nn.Parameter(torch.randn(1,100,1))
        self.unconditioned_cat_embedding = nn.Parameter(torch.randn(1,768,1))
    def get_uncond_batch(self, code_emb):
        unconditioned_batches = torch.zeros((code_emb.shape[0], 1, 1), device=code_emb.device)
        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                               device=code_emb.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_cat_embedding.repeat(code_emb.shape[0], 1, 1),
                                   code_emb)
        return code_emb
    def forward(self, x, t, hint, refer, conditioning_free=False):
        if conditioning_free:
            hint = self.unconditioned_cat_embedding.repeat(x.shape[0], 1, x.shape[-1])
        else:
            if self.training:
                hint = self.get_uncond_batch(hint)
            hint = F.interpolate(hint, size=x.shape[-1], mode='nearest')
        refer_cross = self.refer_enc(refer)
        refer_self = self.refer_model(refer, timesteps = t, context = refer_cross)
        eps = self.base_model(x, timesteps=t, context=refer_cross, hint=hint, control=refer_self)
        return eps
