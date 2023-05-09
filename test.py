import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import torchaudio
from model import NaturalSpeech2, TextEncoder, F0Predictor, Diffusion_Encoder, num_to_groups
from audiolm_pytorch import SoundStream, EncodecWrapper
from dataset import NS2VCDataset,TextAudioCollate
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
import torchaudio.transforms as T



if __name__ == '__main__':
    cfg = json.load(open('config.json'))


    collate_fn = TextAudioCollate()
    codec = EncodecWrapper()
    ds = NS2VCDataset(cfg, codec)
    dl = DataLoader(ds, batch_size = cfg['train']['train_batch_size'], shuffle = True, pin_memory = True, num_workers = 0, collate_fn = collate_fn)
    # c, f0, codes, audio, uv = ds[0]
    # print(c.shape, f0.shape, codes.shape, audio.shape, uv.shape)
    # c_padded, refer_padded, f0_padded, codes_padded, wav_padded, lengths, refer_lengths, uv_padded = next(iter(dl))
    # print(lengths)
    # print(refer_lengths)
    # print(c_padded.shape, refer_padded.shape, f0_padded.shape, codes_padded.shape, wav_padded.shape, lengths.shape, refer_lengths.shape, uv_padded.shape)
    # print(c_padded.shape, codes_padded.shape)
    # print(c_padded[0][0], codes_padded[0][0])
    data = next(iter(dl))
    model = NaturalSpeech2(cfg)
    out = model(data, codec)
    # out.backward()

    # c_padded, refer_padded, f0_padded, codes_padded, wav_padded, lengths, refer_lengths, uv_padded = next(iter(dl))
    # # c_padded refer_padded
    # c = c_padded
    # refer = refer_padded
    # f0 = f0_padded
    # uv = uv_padded
    # codec = EncodecWrapper()
    # with torch.no_grad():
    #     batches = num_to_groups(1, 1)
    #     all_samples_list = list(map(lambda n: model.sample(c, refer, f0, uv, codec, batch_size=n), batches))    
    # all_samples = torch.cat(all_samples_list, dim = 0)
    # torchaudio.save(f'sample.wav', all_samples, 24000)
    # print(lengths)
    # print(refer_lengths)
    


    # phoneme_encoder = TextEncoder(**cfg['phoneme_encoder'])
    # f0_predictor = F0Predictor(**cfg['f0_predictor'])
    # prompt_encoder = TextEncoder(**cfg['prompt_encoder'])
    # diff_model = Diffusion_Encoder(**cfg['diffusion_encoder'])
    # audio_prompt = torch.randn(3, 256, 80)
    # contentvec = torch.randn(3, 256, 200)
    # f0 = torch.randint(1,100,(3, 200))
    # noised_audio = torch.randn(3, 512, 200)
    # times = torch.randn(3)
    # audio_prompt_length = torch.tensor([3, 4, 5])
    # contentvec_length = torch.tensor([3, 4, 5])
    # #ok
    # audio_prompt = prompt_encoder(audio_prompt,audio_prompt_length)
    # #ok
    # f0_pred = f0_predictor(contentvec, audio_prompt, contentvec_length, audio_prompt_length)
    # #ok
    # content = phoneme_encoder(contentvec, contentvec_length,f0)
    # #ok
    # pred = diff_model(
    #     noised_audio,
    #     content, audio_prompt, 
    #     contentvec_length, audio_prompt_length,
    #     times)

# print(codes.shape)#24k 1 128 T2+1



#reconstruction
# codec = EncodecWrapper()
# audio, sr = torchaudio.load('dataset/1.wav')
# audio24k = T.Resample(sr, 24000)(audio)
# torchaudio.save('1_24k.wav', audio24k, 24000)

# codec.eval()
# codes, _, _ = codec(audio24k, return_encoded = True)
# audio = codec.decode(codes).squeeze(0)
# torchaudio.save('1.wav', audio.detach(), 24000)
