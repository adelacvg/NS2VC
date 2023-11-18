from glob import glob
import json
import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
from tqdm import tqdm
import modules.utils as utils
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
        self.cfg=cfg

    def get_audio(self, filename):
        audio, sampling_rate = torchaudio.load(filename)
        audio = T.Resample(sampling_rate, self.sampling_rate)(audio)

        spec = torch.load(filename.replace(".wav", ".mel.pt")).squeeze(0)

        c = torch.load(filename+ ".cvec.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), spec.shape[-1])

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), spec.shape, filename)
        assert abs(audio.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c = spec[:, :lmin], c[:, :lmin]
        audio = audio[:, :lmin * self.hop_length]
        return c.detach(), spec.detach(), audio.detach()

    def __getitem__(self, index):
        index1, index2 = random.sample(range(0, self.__len__()), 2)
        return *self.get_audio(self.audiopaths[index1]), *self.get_audio(self.audiopaths[index2])

    def __len__(self):
        return len(self.audiopaths)


class NS2VCDataset(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audio_path, cfg, all_in_mem: bool = False):
        self.audiopaths = glob(os.path.join(audio_path, "**/*.wav"), recursive=True)
        self.sampling_rate = cfg['data']['sampling_rate']
        self.hop_length = cfg['data']['hop_length']
        self.cfg = cfg
        random.shuffle(self.audiopaths)
        
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p) for p in self.audiopaths]

    def get_audio(self, filename):
        audio, sampling_rate = torchaudio.load(filename)
        audio = T.Resample(sampling_rate, self.sampling_rate)(audio)

        spec = torch.load(filename.replace(".wav", ".mel.pt")).squeeze(0)

        c = torch.load(filename+ ".cvec.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), spec.shape[-1])

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), spec.shape, filename)
        assert abs(audio.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c = spec[:, :lmin], c[:, :lmin]
        audio = audio[:, :lmin * self.hop_length]
        return c.detach(), spec.detach(), audio.detach()

    def random_slice(self, c, spec, audio):
        if spec.shape[1] < 30:
            print("skip too short audio")
            return None
        if spec.shape[1] > 200:
            start = random.randint(0, spec.shape[1]-200)
            end = start + 200
            spec, c = spec[:, start:end], c[:, start:end]
            audio = audio[:, start * self.hop_length : end * self.hop_length]
        len_spec = spec.shape[1]
        clip = random.randint(int(len_spec//3), int(len_spec//3*2))
        if random.random()>0.5:
            refer = spec[:, :clip]
            c = c[:, clip:]
            spec = spec[:, clip:]
            audio = audio[:,clip*self.hop_length:]
        else:
            refer = spec[:, clip:]
            c = c[:, :clip]
            spec = spec[:, :clip]
            audio = audio[:, :clip*self.hop_length]
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
        hop_length = 320
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        # refer, c, spec, audio
        max_refer_len = max([x[0].size(1) for x in batch])
        max_c_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        lengths = torch.LongTensor(len(batch))
        refer_lengths = torch.LongTensor(len(batch))

        contentvec_dim = batch[0][1].shape[0]
        spec_dim = batch[0][2].shape[0]
        c_padded = torch.FloatTensor(len(batch), contentvec_dim, max_c_len+1)
        spec_padded = torch.FloatTensor(len(batch), spec_dim, max_c_len+1)
        refer_padded = torch.FloatTensor(len(batch), spec_dim, max_refer_len+1)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len+1)

        c_padded.zero_()
        spec_padded.zero_()
        refer_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            # refer, c, f0, spec, audio, uv
            len_refer = row[0].size(1)
            len_contentvec = row[1].size(1)
            len_wav = row[3].size(1)

            lengths[i] = len_contentvec
            refer_lengths[i] = len_refer

            refer_padded[i, :, :len_refer] = row[0][:]
            c_padded[i, :, :len_contentvec] = row[1][:]
            spec_padded[i, :, :len_contentvec] = row[2][:]
            wav_padded[i, :, :len_wav] = row[3][:]

        return {
            'cvec':c_padded,
            'refer':refer_padded,
            'spec':spec_padded,
            'wav':wav_padded
        }

if __name__ == "__main__":
    cfg_path='config.json'
    cfg = json.load(open(cfg_path))
    ds = NS2VCDataset(cfg['data']['training_files'], cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=TextAudioCollate())
    min_x = 0
    max_x = 0
    for b in tqdm(dl):
        break
        min_x = min(min_x, b['cvec'].min())
        max_x = max(max_x, b['cvec'].max())
    print(min_x,max_x)
