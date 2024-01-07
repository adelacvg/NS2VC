from glob import glob
import json
import math
import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
import utils
import torchaudio.transforms as T
import random
from tqdm import tqdm

def padding_to_8(x):
    l = x.shape[-1]
    l = (math.floor(l / 8) + 1) * 8
    x = torch.nn.functional.pad(x, (0, l-x.shape[-1]))
    return x

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, audio_path, cfg, codec, all_in_mem: bool = False):
        self.audiopaths = glob(os.path.join(audio_path, "**/*.wav"), recursive=True)
        self.sampling_rate = cfg['data']['sampling_rate']
        self.hop_length = cfg['data']['hop_length']
        random.shuffle(self.audiopaths)
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, filename):
        audio, sampling_rate = torchaudio.load(filename)
        audio = T.Resample(sampling_rate, self.sampling_rate)(audio)

        spec = torch.load(filename.replace(".wav", ".mel.pt")).squeeze(0)

        c = torch.load(filename+ ".cvec.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), spec.shape[-1])

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), filename)
        assert abs(audio.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c = spec[:, :lmin], c[:, :lmin]
        audio = audio[:, :lmin * self.hop_length]

        return c.detach(),  spec.detach(), audio.detach()

    def __getitem__(self, index):
        c,spec,audio = self.get_audio(self.audiopaths[index])
        c_,spec_,audio_ = self.get_audio(self.audiopaths[(index+random.randint(1, 100))%self.__len__()])
        spec,spec_,c = padding_to_8(spec),padding_to_8(spec_),padding_to_8(c)
        return dict(spec=spec, refer=spec_, cvec=c, wav=audio, wav_refer=audio_)

    def __len__(self):
        return len(self.audiopaths)


class NS2VCDataset(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, cfg, all_in_mem: bool = False):
        self.audiopaths = glob(os.path.join(cfg['data']['training_files'], "**/*.wav"), recursive=True)
        self.sampling_rate = cfg['data']['sampling_rate']
        self.hop_length = cfg['data']['hop_length']
        # self.codec = codec

        # random.seed(1234)
        random.shuffle(self.audiopaths)
        
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, filename):
        audio, sampling_rate = torchaudio.load(filename)
        audio = T.Resample(sampling_rate, self.sampling_rate)(audio)

        spec = torch.load(filename.replace(".wav", ".mel.pt")).squeeze(0)

        c = torch.load(filename+ ".cvec.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), spec.shape[-1])

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), filename)
        assert abs(audio.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c = spec[:, :lmin], c[:, :lmin]
        audio = audio[:, :lmin * self.hop_length]
        return c.detach(), spec.detach(), audio.detach()

    def random_slice(self, c, spec, audio):
        if spec.shape[1] < 30:
            print("skip too short audio")
            return None
        if spec.shape[1] > 400:
            start = random.randint(0, spec.shape[1]-400)
            end = start + 400
            spec, c = spec[:, start:end], c[:, start:end]
            audio = audio[:, start * self.hop_length : end * self.hop_length]
        len_spec = spec.shape[1]
        l = random.randint(int(len_spec//3), int(len_spec//3*2))
        u = random.randint(0, len_spec-l)
        v = u + l
        refer = spec[:, u:v]
        c = torch.cat([c[:, :u], c[:, v:]], dim=-1)
        spec = torch.cat([spec[:, :u], spec[:, v:]], dim=-1)
        audio = torch.cat([audio[:, :u * self.hop_length], audio[:, v * self.hop_length:]], dim=-1)
        assert c.shape[1] != 0
        assert refer.shape[1] != 0
        return refer, c, spec, audio

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.random_slice(*self.cache[index])
        else:
            return self.random_slice(*self.get_audio(self.audiopaths[index]))
        # print(1)

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        # refer, c, spec, audio
        max_refer_len = max([x[0].size(1) for x in batch])
        max_refer_len = (math.floor(max_refer_len / 8) + 1) * 8
        max_c_len = max([x[1].size(1) for x in batch])
        max_c_len = (math.floor(max_c_len / 8) + 1) * 8
        max_wav_len = max([x[3].size(1) for x in batch])

        lengths = torch.LongTensor(len(batch))
        refer_lengths = torch.LongTensor(len(batch))

        contentvec_dim = batch[0][1].shape[0]
        spec_dim = batch[0][2].shape[0]
        c_padded = torch.FloatTensor(len(batch), contentvec_dim, max_c_len)
        spec_padded = torch.FloatTensor(len(batch), spec_dim, max_c_len)
        refer_padded = torch.FloatTensor(len(batch), spec_dim, max_refer_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len+1)

        c_padded.zero_()
        spec_padded.zero_()
        refer_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            # refer, c, spec, audio
            len_refer = row[0].size(1)
            len_contentvec = row[1].size(1)
            len_wav = row[3].size(1)

            lengths[i] = len_contentvec
            refer_lengths[i] = len_refer

            refer_padded[i, :, :len_refer] = row[0][:]
            c_padded[i, :, :len_contentvec] = row[1][:]
            spec_padded[i, :, :len_contentvec] = row[2][:]
            wav_padded[i, :, :len_wav] = row[3][:]
        return dict(spec=spec_padded, refer=refer_padded, cvec=c_padded, wav=wav_padded)
        # return c_padded, refer_padded, spec_padded, wav_padded
if __name__=='__main__':
    cfg = json.load(open('config.json'))
    ds = NS2VCDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=TextAudioCollate())
    for b in tqdm(dl):
        break
