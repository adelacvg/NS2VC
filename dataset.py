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


class NS2VCDataset(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, cfg,codec, all_in_mem: bool = False):
        self.audiopaths = glob(os.path.join(cfg['data']['training_files'], "**/*.wav"), recursive=True)
        self.sampling_rate = cfg['data']['sampling_rate']
        self.use_sr = cfg['train']['use_sr']
        self.spec_len = cfg['train']['max_speclen']
        self.hop_length = cfg['data']['hop_length']
        self.codec = codec

        random.seed(1234)
        random.shuffle(self.audiopaths)
        
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, filename):
        audio, sampling_rate = torchaudio.load(filename)
        audio = T.Resample(sampling_rate, self.sampling_rate)(audio)
        # if sampling_rate != self.sampling_rate:
        #     raise ValueError("{} SR doesn't match target {} SR".format(
        #         sampling_rate, self.sampling_rate))

        codes = torch.load(filename.replace(".wav", ".code.pt")).squeeze(0)

        f0 = np.load(filename + ".f0.npy")
        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)

        c = torch.load(filename+ ".soft.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0])


        # print(codes.shape, c.shape, f0.shape, audio.shape, uv.shape)
        lmin = min(c.size(-1), codes.size(-1))
        # print(lmin)
        assert abs(c.size(-1) - codes.size(-1)) < 3, (c.size(-1), codes.size(-1), f0.shape, filename)
        assert abs(audio.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        codes, c, f0, uv = codes[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin]
        audio = audio[:, :lmin * self.hop_length]
        return c, f0, codes, audio, uv

    def random_slice(self, c, f0, codes, audio, uv):
        # if spec.shape[1] < 30:
        #     print("skip too short audio:", filename)
        #     return None
        if codes.shape[1] > 800:
            start = random.randint(0, codes.shape[1]-800)
            end = start + 790
            codes, c, f0, uv = codes[:, start:end], c[:, start:end], f0[start:end], uv[start:end]
            audio = audio[:, start * self.hop_length : end * self.hop_length]

        return c, f0, codes, audio, uv

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
        # print("yes")
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        max_c_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        lengths = torch.LongTensor(len(batch))
        refer_lengths = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_c_len)
        f0_padded = torch.FloatTensor(len(batch), max_c_len)
        codes_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        refer_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        uv_padded = torch.FloatTensor(len(batch), max_c_len)

        c_padded.zero_()
        codes_padded.zero_()
        refer_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            len_raw = row[0].size(1)
            u,v = sorted(random.sample(range(1,len_raw-1), 2))

            lengths[i] = len_raw - (v-u+1)
            refer_lengths[i] = v-u+1
            # print(u,v)
            refer_padded[i, :, :v-u+1] = row[2][:,u:v+1]

            c = row[0]
            # print(c[:,v+1:].shape, lengths[i])
            c_padded[i, :, :u] = c[:,:u]
            c_padded[i, :, u:u+len_raw-v-1] = c[:,v+1:]

            f0 = row[1]
            f0_padded[i, :u] = f0[:u]
            f0_padded[i, u:u+len_raw-v-1] = f0[v+1:]
            # f0_padded[i, :f0.size(0)] = f0

            codes = row[2]
            codes_padded[i, :, :u] = codes[:,:u]
            codes_padded[i, :, u:u+len_raw-v-1] = codes[:,v+1:]
            # print(u+len_raw-v-1)
            # codes_padded[i, :, :codes.size(1)] = codes

            # audio = audio[:, :lmin * self.hop_length]
            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav

            uv = row[4]
            uv_padded[i, :u] = uv[:u]
            uv_padded[i, u:u+len_raw-v-1] = uv[v+1:]
            # uv_padded[i, :uv.size(0)] = uv
        

        # print(c_padded.shape, f0_padded.shape, codes_padded.shape, wav_padded.shape, uv_padded.shape)
        return c_padded, refer_padded, f0_padded, codes_padded, wav_padded, lengths, refer_lengths, uv_padded
