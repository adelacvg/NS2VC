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

        random.shuffle(self.audiopaths)
        
        self.all_in_mem = all_in_mem
        self.cleaners = []
        if cfg['data']['language'] == 'en':
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
        assert abs(codes.size(-1) - sum(duration)) < 3, (codes.size(-1), sum(duration), filename)
        assert abs(audio.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        assert phone.shape[0] == duration.shape[0]
        codes, f0, uv = codes[:, :lmin], f0[:lmin], uv[:lmin]
        audio = audio[:, :lmin * self.hop_length]
        if sum(duration) > lmin:
            duration[-1] = lmin - sum(duration[:-1])
        return f0.detach(), codes.detach(), audio.detach(), uv.detach(), phone.detach(), duration.detach()

    def random_slice(self, f0, codes, audio, uv, phone, duration):
        if phone.shape[0] < 3:
            print("skip too short audio")
            return None
        if phone.shape[0] > 30:
            start = random.randint(0, phone.shape[0]-30)
            end = start + 30
            start_frame = sum(duration[:start])
            end_frame = sum(duration[:end])
            phone, duration = phone[start:end], duration[start:end]
            f0 = f0[start_frame:end_frame]
            codes = codes[:, start_frame:end_frame]
            uv = uv[start_frame:end_frame]
            audio = audio[:, start_frame*self.hop_length:end_frame*self.hop_length]
        len_phoneme = phone.shape[0]
        l = random.randint((len_phoneme//3), (len_phoneme//3*2))
        u = random.randint(0, len_phoneme-l)
        v = u + l
        s = sum(duration[:u])
        e = sum(duration[:v])
        refer = codes[:,s:e]
        f0 = torch.cat([f0[:s], f0[e:]])
        codes = torch.cat([codes[:,:s], codes[:,e:]], dim=1)
        audio = torch.cat([audio[:,:s*self.hop_length], audio[:,e*self.hop_length:]], dim=1)
        uv = torch.cat([uv[:s], uv[e:]])
        phone = torch.cat([phone[:u], phone[v:]])
        duration = torch.cat([duration[:u], duration[v:]])
        assert refer.shape[1] != 0
        assert audio.shape[1] != 0

        return refer, f0, codes, audio, uv, phone, duration

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.random_slice(*self.cache[index])
        else:
            return self.random_slice(*self.get_audio(self.audiopaths[index]))

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]#B T C 

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[2].shape[1] for x in batch]),
            dim=0, descending=True)
        
        max_refer_len = max([x[0].size(1) for x in batch])
        max_f0_len = max([x[1].size(0) for x in batch])
        max_code_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])
        max_text_len = max([x[5].size(0) for x in batch])
        code_dim = batch[0][2].shape[0]
        assert(max_f0_len == max_code_len)

        code_lengths = torch.LongTensor(len(batch))
        refer_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))

        f0_padded = torch.FloatTensor(len(batch), max_code_len+1)
        codes_padded = torch.FloatTensor(len(batch), code_dim, max_code_len+1)
        refer_padded = torch.FloatTensor(len(batch), code_dim, max_refer_len+1)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len+1)
        uv_padded = torch.FloatTensor(len(batch), max_code_len+1)

        duration_padded = torch.LongTensor(len(batch), max_text_len+1)
        phoneme_padded = torch.LongTensor(len(batch), max_text_len+1)

        codes_padded.zero_()
        refer_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()
        phoneme_padded.zero_()
        duration_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            len_code = row[2].size(1)
            len_text = row[5].size(0)
            len_refer = row[0].size(1)
            len_wav = row[3].size(1)
            code_lengths[i] = len_code
            refer_lengths[i] = len_refer
            text_lengths[i] = len_text
            # refer, f0, codes, audio, uv, phone, duration
            refer_padded[i, :, :len_refer] = row[0][:]
            f0_padded[i, :len_code] = row[1][:]
            codes_padded[i, :, :len_code] = row[2][:]
            wav_padded[i, :, :len_wav] = row[3][:]
            uv_padded[i, :len_code] = row[4][:]
            phoneme_padded[i, :len_text] = row[5][:]
            duration_padded[i, :len_text] = row[6][:]

        return refer_padded, f0_padded, codes_padded, \
        wav_padded, code_lengths, refer_lengths, text_lengths,\
        uv_padded, phoneme_padded, duration_padded
