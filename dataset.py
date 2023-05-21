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

    def __init__(self, cfg,codec, all_in_mem: bool = True):
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
        phone_path = filename + ".phone.txt"
        with open(phone_path, "r") as f:
            text = f.readline()
        phone = np.array(text_to_sequence(text, self.cleaners))
        phone = torch.LongTensor(phone)

        duration = np.load(filename + ".duration.npy")
        duration = torch.LongTensor(duration)

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
        # print(c.size(-1), codes.size(-1), sum(duration))
        lmin = min(c.size(-1), codes.size(-1), sum(duration))
        # print(lmin)
        assert abs(c.size(-1) - codes.size(-1)) < 3, (c.size(-1), codes.size(-1), f0.shape, filename)
        # assert abs(audio.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        codes, c, f0, uv = codes[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin]
        audio = audio[:, :lmin * self.hop_length]
        if sum(duration) > lmin:
            duration[-1] = lmin - sum(duration[:-1])
            if duration[-1]==0:
                duration = duration[:-1]
        # print(phone)
        # print(duration)
        return c.detach(), f0.detach(), codes.detach(), audio.detach(), uv.detach(), phone.detach(), duration.detach()

    def random_slice(self, c, f0, codes, audio, uv, phone, duration):
        # if spec.shape[1] < 30:
        #     print("skip too short audio:", filename)
        #     return None
        # if codes.shape[1] > 800:
        #     start = random.randint(0, codes.shape[1]-800)
        #     end = start + 790
        #     codes, c, f0, uv = codes[:, start:end], c[:, start:end], f0[start:end], uv[start:end]
        #     audio = audio[:, start * self.hop_length : end * self.hop_length]

        return c, f0, codes, audio, uv, phone, duration

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.random_slice(*self.cache[index])
        else:
            return self.random_slice(*self.get_audio(self.audiopaths[index]))

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:

    def __call__(self, batch):
        # print("yes")
        hop_length = 320
        batch = [b for b in batch if b is not None]#B T C 
        #95 79
        #13 12

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        max_c_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])
        max_text_len = max([x[5].size(0) for x in batch])

        lengths = torch.LongTensor(len(batch))
        refer_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_c_len)
        f0_padded = torch.FloatTensor(len(batch), max_c_len)
        codes_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        refer_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        uv_padded = torch.FloatTensor(len(batch), max_c_len)

        duration_padded = torch.LongTensor(len(batch), max_text_len)
        phoneme_padded = torch.LongTensor(len(batch), max_text_len)

        c_padded.zero_()
        codes_padded.zero_()
        refer_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()
        phoneme_padded.zero_()
        duration_padded.zero_()

        # for i in range(len(ids_sorted_decreasing)):
        #     row = batch[ids_sorted_decreasing[i]]
        #     u,v = sorted(random.sample(range(0,row[0].size(1)), 2))
        #     c_padded[i, :, :v-u+1] = row[0][:,u:v+1]
        #     f0_padded[i, :v-u+1] = row[1][u:v+1]
        #     codes_padded[i, :, :v-u+1] = row[2][:,u:v+1]
        #     refer_padded[i, :, :row[0].size(1)] = row[2]
        #     wav_padded[i, :, :row[3].size(1)] = row[3]
        #     uv_padded[i, :v-u+1] = row[4][u:v+1]
        #     lengths[i] = v-u+1
        #     refer_lengths[i] = row[0].size(1)

            # c_padded[i, :, :row[0].size(1)] = row[0]
            # f0_padded[i, :row[0].size(1)] = row[1]
            # codes_padded[i, :, :row[0].size(1)] = row[2]
            # refer_padded[i, :, :row[0].size(1)] = row[2]
            # wav_padded[i, :, :row[3].size(1)] = row[3]
            # uv_padded[i, :row[0].size(1)] = row[4]
            # lengths[i] = row[0].size(1)
            # refer_lengths[i] = row[0].size(1)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]


            len_raw = row[0].size(1)
            len_phoneme = row[5].size(0)
            l = random.randint(1, len_phoneme-1)
            u = random.randint(0, len_phoneme-l)
            v = u + l - 1

            text_lengths[i] = len_phoneme - l
            refer_lengths[i] = sum(row[6][u:v+1])
            lengths[i] = len_raw - refer_lengths[i]
            # print(refer_lengths[i], lengths[i])
            # lengths[i] = len_raw - (v-u+1)
            # refer_lengths[i] = v-u+1
            # if refer_lengths[i] > lengths[i]:
            # refer_padded[i, :, :v-u+1] = row[2][:,u:v+1]
            # print(len_raw)
            # print(row[6])
            # print(u,v)
            s = sum(row[6][:u])
            e = sum(row[6][:v+1])
            # print(s,e)
            # print(refer_lengths[i], lengths[i])
            refer_padded[i, :, :refer_lengths[i]] = row[2][:,s:e]

            c = row[0]
            c_padded[i, :, :s] = c[:,:s]
            c_padded[i, :, s:s+len_raw-e] = c[:,e:]

            f0 = row[1]
            f0_padded[i, :s] = f0[:s]
            f0_padded[i, s:s+len_raw-e] = f0[e:]
            # print(f0_padded[i].shape)

            codes = row[2]
            codes_padded[i, :, :s] = codes[:,:s]
            codes_padded[i, :, s:s+len_raw-e] = codes[:,e:]

            wav = row[3]
            wav_padded[i, :, :s*hop_length] = wav[:,:s*hop_length]
            wav_padded[i, :, s*hop_length:s*hop_length+len_raw*hop_length-e*hop_length] = wav[:,e*hop_length:]

            uv = row[4]
            uv_padded[i, :s] = uv[:s]
            uv_padded[i, s:s+len_raw-e] = uv[e:]

            phoneme = row[5]
            phoneme_padded[i, :u] = phoneme[:u]
            phoneme_padded[i, u:u+len_phoneme-v-1] = phoneme[v+1:]

            duration = row[6]
            duration_padded[i, :u] = duration[:u]
            duration_padded[i, u:u+len_phoneme-v-1] = duration[v+1:]
            # print(sum(duration_padded[i]))
        #     else:
        #         lengths[i], refer_lengths[i] = refer_lengths[i], lengths[i]
        #         refer_padded[i,:,:u] = row[2][:,:u]
        #         refer_padded[i,:,u:u+len_raw-v-1] = row[2][:,v+1:]
        #         c_padded[i,:,:v-u+1] = row[0][:,u:v+1]
        #         f0_padded[i,:v-u+1] = row[1][u:v+1]
        #         codes_padded[i,:,:v-u+1] = row[2][:,u:v+1]
        #         wav = row[3]
        #         wav_padded[i, :, :wav.size(1)] = wav
        #         uv_padded[i, :v-u+1] = row[4][u:v+1]

        

        # print(c_padded.shape, f0_padded.shape, codes_padded.shape, wav_padded.shape, uv_padded.shape)
        return c_padded, refer_padded, f0_padded, codes_padded, \
        wav_padded, lengths, refer_lengths, text_lengths, uv_padded, phoneme_padded, duration_padded
