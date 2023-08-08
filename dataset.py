from glob import glob
import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
import utils
import torchaudio.transforms as T
import random


"""Multi speaker version"""

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

        spec = torch.load(filename.replace(".wav", ".spec.pt")).squeeze(0)

        f0 = np.load(filename + ".f0.npy")
        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)

        c = torch.load(filename+ ".soft.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0])

        prosody = torch.load(filename.replace(".wav", ".prosody.pt")).squeeze(0)

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), f0.shape, filename)
        assert abs(audio.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv, prosody = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin], prosody
        audio = audio[:, :lmin * self.hop_length]
        return c.detach(), f0.detach(), spec.detach(), audio.detach(), uv.detach(), prosody.detach()

    def __getitem__(self, index):
        return *self.get_audio(self.audiopaths[index]), *self.get_audio(self.audiopaths[(index+4)%self.__len__()])

    def __len__(self):
        return len(self.audiopaths)


class NS2VCDataset(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audio_path, cfg, codec, all_in_mem: bool = False):
        self.audiopaths = glob(os.path.join(audio_path, "**/*.wav"), recursive=True)
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

        spec = torch.load(filename.replace(".wav", ".spec.pt")).squeeze(0)

        f0 = np.load(filename + ".f0.npy")
        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)

        c = torch.load(filename+ ".soft.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0])

        prosody = torch.load(filename.replace(".wav", ".prosody.pt")).squeeze(0)

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), f0.shape, filename)
        assert abs(audio.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv, prosody = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin], prosody[:, :lmin]
        audio = audio[:, :lmin * self.hop_length]
        return c.detach(), f0.detach(), spec.detach(), audio.detach(), uv.detach(), prosody.detach()

    def random_slice(self, c, f0, spec, audio, uv, prosody):
        if spec.shape[1] < 30:
            print("skip too short audio")
            return None
        if spec.shape[1] > 400:
            start = random.randint(0, spec.shape[1]-400)
            end = start + 400
            spec, c, f0, uv, prosody = spec[:, start:end], c[:, start:end], f0[start:end], uv[start:end], prosody[:, start:end]
            audio = audio[:, start * self.hop_length : end * self.hop_length]
        len_spec = spec.shape[1]
        l = random.randint(int(len_spec//3), int(len_spec//3*2))
        u = random.randint(0, len_spec-l)
        v = u + l
        refer = spec[:, u:v]
        c = torch.cat([c[:, :u], c[:, v:]], dim=-1)
        f0 = torch.cat([f0[:u], f0[v:]], dim=-1)
        spec = torch.cat([spec[:, :u], spec[:, v:]], dim=-1)
        prosody = torch.cat([prosody[:, :u], prosody[:, v:]], dim=-1)
        uv = torch.cat([uv[:u], uv[v:]], dim=-1)
        audio = torch.cat([audio[:, :u * self.hop_length], audio[:, v * self.hop_length:]], dim=-1)
        assert c.shape[1] != 0
        assert refer.shape[1] != 0
        return refer, c, f0, spec, audio, uv, prosody

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
        hop_length = 320
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        # refer, c, f0, spec, audio, uv
        max_refer_len = max([x[0].size(1) for x in batch])
        max_c_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[4].size(1) for x in batch])

        lengths = torch.LongTensor(len(batch))
        refer_lengths = torch.LongTensor(len(batch))

        contentvec_dim = batch[0][1].shape[0]
        spec_dim = batch[0][3].shape[0]
        c_padded = torch.FloatTensor(len(batch), contentvec_dim, max_c_len+1)
        f0_padded = torch.FloatTensor(len(batch), max_c_len+1)
        spec_padded = torch.FloatTensor(len(batch), spec_dim, max_c_len+1)
        refer_padded = torch.FloatTensor(len(batch), spec_dim, max_refer_len+1)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len+1)
        uv_padded = torch.FloatTensor(len(batch), max_c_len+1)
        prosody_padded = torch.FloatTensor(len(batch), 400, max_c_len+1)

        c_padded.zero_()
        spec_padded.zero_()
        refer_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()
        prosody_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            # refer, c, f0, spec, audio, uv, prosody
            len_refer = row[0].size(1)
            len_contentvec = row[1].size(1)
            len_wav = row[4].size(1)

            lengths[i] = len_contentvec
            refer_lengths[i] = len_refer

            refer_padded[i, :, :len_refer] = row[0][:]
            c_padded[i, :, :len_contentvec] = row[1][:]
            f0_padded[i, :len_contentvec] = row[2][:]
            spec_padded[i, :, :len_contentvec] = row[3][:]
            wav_padded[i, :, :len_wav] = row[4][:]
            uv_padded[i, :len_contentvec] = row[5][:]
            prosody_padded[i, :, :len_contentvec] = row[6][:]

        return c_padded, refer_padded, f0_padded, spec_padded, wav_padded, lengths, refer_lengths, uv_padded, prosody_padded
