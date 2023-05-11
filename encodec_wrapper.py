from functools import reduce
from einops import rearrange, pack, unpack

import torch
from torch import nn

from vector_quantize_pytorch import ResidualVQ

from encodec import EncodecModel
from encodec.utils import _linear_overlap_add

class EncodecWrapper(nn.Module):
    """
    Support pretrained 24kHz Encodec by Meta AI, if you want to skip training SoundStream.

    TODO:
    - see if we need to keep the scaled version and somehow persist the scale factors for when we need to decode? Right
        now I'm just setting self.model.normalize = False to sidestep all of that
    - see if we can use the 48kHz model, which is specifically for music. Right now we're using the 24kHz model because
        that's what was used in MusicLM and avoids any resampling issues.
    -

    """
    def __init__(
        self,
        target_sample_hz = 24000,
        strides = (2, 4, 5, 8),
        num_quantizers = 8,
    ):
        super().__init__()
        # Instantiate a pretrained EnCodec model
        self.model = EncodecModel.encodec_model_24khz()
        self.model.normalize = False # this means we don't need to scale codes e.g. when running model.encode(wav)

        # bandwidth affects num quantizers used: https://github.com/facebookresearch/encodec/pull/41
        self.model.set_target_bandwidth(6.0)
        assert num_quantizers == 8, "assuming 8 quantizers for now, see bandwidth comment above"

        # Fields that SoundStream has that get used externally. We replicate them here.
        self.target_sample_hz = target_sample_hz
        assert self.target_sample_hz == 24000, "haven't done anything with non-24kHz yet"

        self.codebook_dim = 128
        self.num_quantizers = num_quantizers
        self.strides = strides # used in seq_len_multiple_of

    @property
    def seq_len_multiple_of(self):
        return reduce(lambda x, y: x * y, self.strides)

    def forward(
        self,
        x,
        return_encoded = False,
        **kwargs
    ):

        x, ps = pack([x], '* n')

        # kwargs for stuff like return_encoded=True, which SoundStream uses but Encodec doesn't
        assert not self.model.training, "Encodec is pretrained and should never be called outside eval mode."
        # Unlike in the Encodec sample code in its README, x has already been resampled so we don't need to call
        # convert_audio and unsqueeze. The convert_audio function also doesn't play nicely with batches.

        # b = batch, t = timesteps, 1 channel for the 24kHz model, 2 channels for the 48kHz model
        wav = rearrange(x, f'b t -> b {self.model.channels} t')

        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
        # encoded_frames is a list of (frame, scale) tuples. Scale is a scalar but we don't use it. Frame is a tensor
        # of shape [batch, num_quantizers, num_samples_per_frame]. We want to concatenate the frames to get all the
        # timesteps concatenated.
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=1)  # [batch, num_quantizers, timesteps]
        # transformer code that uses codec expects codes to be [batch, timesteps, num_quantizers]
        codes = rearrange(codes, 'b q n -> b n q')  # result: [batch, timesteps, num_quantizers]
        # in original soundstream, is x, indices, commit_loss. But we only use indices in eval mode, so just keep that.

        # allow for returning of sum of quantized embeddings

        emb = None
        if return_encoded:
            emb = self.get_emb_from_indices(codes)

        emb, = unpack(emb, ps, '* n c')
        codes, = unpack(codes, ps, '* n q')

        return emb, codes, None
    def decode(self, emb):
        emb = rearrange(emb, 'b n c -> b c n')
        return self.model.decoder(emb)
