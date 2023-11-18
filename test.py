import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import torchaudio
# from model import NaturalSpeech2, F0Predictor, Diffusion_Encoder, encode
from dataset import NS2VCDataset, TextAudioCollate
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
import torchaudio.transforms as T
# from model import rvq_ce_loss



# if __name__ == '__main__':
    # cfg = json.load(open('config.json'))

    # collate_fn = TextAudioCollate()
    # codec = EncodecWrapper()
    # ds = NS2VCDataset(cfg, codec)
    # dl = DataLoader(ds, batch_size = cfg['train']['train_batch_size'], shuffle = True, pin_memory = True, num_workers = 0, collate_fn = collate_fn)
    # # c_padded, refer_padded, f0_padded, codes_padded, wav_padded, lengths, refer_lengths, uv_padded = next(iter(dl))
    # data = next(iter(dl))
    # model = NaturalSpeech2(cfg)
    # out = model(data, codec)

    # print(c_padded.shape, refer_padded.shape, f0_padded.shape, codes_padded.shape, wav_padded.shape, lengths.shape, refer_lengths.shape, uv_padded.shape)
    # torch.Size([8, 256, 276]) torch.Size([8, 128, 276]) torch.Size([8, 276]) torch.Size([8, 128, 276]) torch.Size([8, 1, 88320]) torch.Size([8]) torch.Size([8]) torch.Size([8, 276])

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

# codec = EncodecWrapper()
# gt = torch.randn(4, 128, 276)
# pred = torch.randn(4, 128, 276)
# _, indices, _, quantized_list = encode(gt,8,codec)
# n_q=8
# loss = rvq_ce_loss(gt.unsqueeze(0)-quantized_list, indices, codec, n_q)
# print(loss)
# loss = rvq_ce_loss(pred.unsqueeze(0)-quantized_list, indices, codec, n_q)
# print(loss)
# wav,sr = torchaudio.load('/home/hyc/val_dataset/common_voice_zh-CN_37110506.mp3')
# wav24k = T.Resample(sr, 24000)(wav)
# spec_process = torchaudio.transforms.MelSpectrogram(
#     sample_rate=24000,
#     n_fft=1024,
#     hop_length=256,
#     n_mels=100,
#     center=True,
#     power=1,
# )
# spec = spec_process(wav24k)# 1 100 T
# spec = torch.log(torch.clip(spec, min=1e-7))
# print(spec)
# print(spec.shape)

# prosody_process = torchaudio.transforms.MelSpectrogram(
#     sample_rate=24000,
#     n_fft=8192,
#     hop_length=4096,
#     n_mels=400,
#     center=True,
#     power=1,
# )
# prosody = prosody_process(wav24k)# 1 400 T
# prosody = torch.log(torch.clip(prosody, min=1e-7))
# prosody = torch.repeat_interleave(prosody, 16, dim=2)
# prosody[:,:,16:] = (prosody[:,:,16:] + prosody[:,:,:-16]) / 2
# print(prosody)
# print(prosody.shape)

# import diffusers
# from diffusers import UNet1DModel,UNet2DConditionModel
# from train import NaturalSpeech2

# from unet1d import UNet1DConditionModel

# a = torch.randn(4, 20, 10)
# lengths = torch.tensor([10, 9, 8, 7])
# print(torch.arange(10))
# print(torch.arange(10).expand(4, 20, 10))
# mask = torch.arange(10).expand(4, 20, 10) >= lengths.unsqueeze(1).unsqueeze(1)
# a = a.masked_fill(mask,0)
# print(a)

# unet2d = UNet2DConditionModel(
#     block_out_channels=(1,2,4,4),
#     norm_num_groups=1,
#     cross_attention_dim=16,
#     attention_head_dim=1,
# )
# in_img = torch.randn(1,4,16,16)
# cond = torch.randn(1,4,16)
# out = unet2d(in_img, 3, cond)
# print(out.sample.shape)

# unet1d = UNet1DConditionModel(
#     in_channels=1,
#     out_channels=1,
#     block_out_channels=(4,8,8,8),
#     norm_num_groups=2,
#     cross_attention_dim=16,
#     attention_head_dim=2,
# )
# audio = torch.randn(1,1,17)
# cond = torch.randn(1,20,16)
# out = unet1d(audio, 3, cond)
# print(out.sample.shape)

# from nsf_hifigan.models import load_model
# import modules.utils as utils
# wav, sr = torchaudio.load('raw/test1.wav')
# wav = T.Resample(sr, 44100)(wav)
# spec_process = torchaudio.transforms.MelSpectrogram(
#         sample_rate=44100,
#         n_fft=2048,
#         hop_length=512,
#         n_mels=128,
#         center=True,
#         power=1,
#     )

# f0 = utils.compute_f0_dio(
#         wav.cpu().numpy()[0], sampling_rate=44100, hop_length=512
#     )
# f0 = torch.Tensor(f0)
# mel = spec_process(wav)
# mel = torch.log(torch.clip(mel, min=1e-7))
# device = 'cuda'
# vocoder = load_model('nsf_hifigan/model',device=device)[0]
# mel = mel.to(device)
# f0 = f0.to(device)
# length = min(mel.shape[2],f0.shape[0])
# mel = mel[:,:,:length]
# f0 = f0[:length]
# wav = vocoder(mel, f0).cpu().squeeze(0)
# torchaudio.save('recon.wav', wav, 44100)

import dac
from audiotools import AudioSignal

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

model.to('cpu')
model.eval()

# Load audio signal file
signal = AudioSignal('/home/hyc/NS2VC/dataset/SSB0011/SSB00110001.wav')

signal.to(model.device)

out = []
signal = signal.cpu()
x = model.compress(signal)

# Save and load to and from disk
x.save("compressed.dac")
x = dac.DACFile.load("compressed.dac")

# Decompress it back to an AudioSignal
y = model.decompress(x)

y.write('output.wav')



