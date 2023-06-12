from glob import glob
import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
from text import text_to_sequence
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
        self.hop_length = cfg['data']['hop_length']
        self.codec = codec

        # random.seed(1234)
        random.shuffle(self.audiopaths)
        
        self.all_in_mem = all_in_mem
        self.cleaners = ["english_cleaners"]
        if self.all_in_mem:
            self.cache = [self.get_audio(p) for p in self.audiopaths]

    def get_audio(self, filename):
        audio, sampling_rate = torchaudio.load(filename)
        audio = T.Resample(sampling_rate, self.sampling_rate)(audio)

        phone_path = filename + ".phone.txt"
        with open(phone_path, "r") as f:
            text = f.readline()
        phone = np.array(text_to_sequence(text, self.cleaners))
        phone = torch.LongTensor(phone)

        duration = np.load(filename + ".duration.npy")
        duration = torch.LongTensor(duration)

        codes = torch.load(filename.replace(".wav", ".code.pt")).squeeze(0)

        f0 = np.load(filename + ".f0.npy")
        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)

        lmin = min(f0.size(-1), codes.size(-1), sum(duration))
        assert abs(f0.size(-1) - codes.size(-1)) < 3, (codes.size(-1), f0.shape, filename)
        assert abs(audio.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        codes, f0, uv = codes[:, :lmin], f0[:lmin], uv[:lmin]
        audio = audio[:, :lmin * self.hop_length]
        if sum(duration) > lmin:
            duration[-1] = lmin - sum(duration[:-1])
        return f0.detach(), codes.detach(), audio.detach(), uv.detach(), phone.detach(), duration.detach()

    def random_slice(self, f0, codes, audio, uv, phone, duration):
        if f0.shape[0] < 30:
            print("skip too short audio")
            return None
        # if f0.shape[0] > 800:
        #     start = random.randint(0, codes.shape[1]-800)
        #     end = start + 790
        #     codes, f0, uv = codes[:, start:end], f0[start:end], uv[start:end]
        #     audio = audio[:, start * self.hop_length : end * self.hop_length]

        return f0, codes, audio, uv, phone, duration

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.random_slice(*self.cache[index])
        else:
            return self.random_slice(*self.get_audio(self.audiopaths[index]))

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:

    def __call__(self, batch):
        hop_length = 320
        batch = [b for b in batch if b is not None]#B T C 

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[0] for x in batch]),
            dim=0, descending=True)
        
        max_f0_len = max([x[0].size(0) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        max_text_len = max([x[4].size(0) for x in batch])

        lengths = torch.LongTensor(len(batch))
        refer_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))

        f0_padded = torch.FloatTensor(len(batch), max_f0_len)
        codes_padded = torch.FloatTensor(len(batch), batch[0][1].shape[0], max_f0_len)
        refer_padded = torch.FloatTensor(len(batch), batch[0][1].shape[0], max_f0_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        uv_padded = torch.FloatTensor(len(batch), max_f0_len)

        duration_padded = torch.LongTensor(len(batch), max_text_len)
        phoneme_padded = torch.LongTensor(len(batch), max_text_len)

        codes_padded.zero_()
        refer_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()
        phoneme_padded.zero_()
        duration_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]


            len_raw = row[0].size(0)
            len_phoneme = row[4].size(0)

            l = random.randint((len_phoneme//3), (len_phoneme//3*2))
            u = random.randint(0, len_phoneme-l)
            v = u + l - 1

            text_lengths[i] = len_phoneme - l
            refer_lengths[i] = sum(row[5][u:v+1])
            lengths[i] = len_raw - refer_lengths[i]

            s = sum(row[5][:u])
            e = sum(row[5][:v+1])
            refer_padded[i, :, :refer_lengths[i]] = row[1][:,s:e]

            f0 = row[0]
            f0_padded[i, :s] = f0[:s]
            f0_padded[i, s:s+len_raw-e] = f0[e:]

            codes = row[1]
            codes_padded[i, :, :s] = codes[:,:s]
            codes_padded[i, :, s:s+len_raw-e] = codes[:,e:]

            wav = row[2]
            wav_padded[i, :, :s*hop_length] = wav[:,:s*hop_length]
            wav_padded[i, :, s*hop_length:s*hop_length+len_raw*hop_length-e*hop_length] = wav[:,e*hop_length:]

            uv = row[3]
            uv_padded[i, :s] = uv[:s]
            uv_padded[i, s:s+len_raw-e] = uv[e:]

            phoneme = row[4]
            phoneme_padded[i, :u] = phoneme[:u]
            phoneme_padded[i, u:u+len_phoneme-v-1] = phoneme[v+1:]

            duration = row[5]
            duration_padded[i, :u] = duration[:u]
            duration_padded[i, u:u+len_phoneme-v-1] = duration[v+1:]

        return refer_padded, f0_padded, codes_padded, \
        wav_padded, lengths, refer_lengths, text_lengths, uv_padded, phoneme_padded, duration_padded
