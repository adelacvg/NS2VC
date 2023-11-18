import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
import functools
import math
import os
import argparse
from random import shuffle
import torchaudio
import torchaudio.transforms as T
import dac
from audiotools import AudioSignal

import torch
from glob import glob
from tqdm import tqdm
from modules.vocoder import Vocoder

import modules.utils as utils
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import numpy as np

hps = utils.get_hparams_from_file("config.json")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
in_dir = ""

def process_batch(filenames):
    print("Loading hubert for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmodel = utils.get_hubert_model().to(device)
    # codec = EncodecWrapper()
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        wav, sr = torchaudio.load(filename)
        if wav.shape[0] > 1:  # to mono
            wav = wav[0].unsqueeze(0)
        if wav.shape[1]<2*sampling_rate:
            continue
        if wav.shape[1]>10*sampling_rate:
            wav = wav[:,:10*sampling_rate]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav = wav.to(device)
        wav16k = T.Resample(sr, 16000).to(device)(wav)
        wav24k = T.Resample(sr, 24000).to(device)(wav)
        wav44k = T.Resample(sr, 44100).to(device)(wav)
        filename = filename.replace(in_dir, in_dir+"_processed").replace('.mp3','.wav').replace('.flac','.wav')
        wav_resample_path = filename
        if not os.path.exists(os.path.dirname(wav_resample_path)):
            os.makedirs(os.path.dirname(wav_resample_path))
        cvec_path = filename + ".cvec.pt"
        c = utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k[0])
        torch.save(c.cpu(), cvec_path)

        if hps.train.vocoder == 'vocos':
            torchaudio.save(wav_resample_path, wav24k.cpu(), 24000)
            spec_path = filename.replace(".wav", ".mel.pt")
            spec_process = torchaudio.transforms.MelSpectrogram(
                sample_rate=24000,
                n_fft=4*hop_length,
                hop_length=hop_length,
                n_mels=100,
                center=True,
                power=1,
            ).to(device)
            spec = spec_process(wav24k.to(device))# 1 100 T
            spec = torch.log(torch.clip(spec, min=1e-5))
            torch.save(spec.detach().cpu(), spec_path)
            
        elif hps.train.vocoder == 'nsf-hifigan':
            #todo not usable now
            pass
        elif hps.train.vocoder == 'dac':
            signal = AudioSignal(filename)
            model_path = dac.utils.download(model_type="44khz")
            model = dac.DAC.load(model_path)

            model.to('cuda')
            signal = signal.cpu()
            x = model.compress(signal)
            dac_path = filename.replace(".wav", ".dac")
            x.save(dac_path)
        else:
            print('not supported vocoder')


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="val_dataset", help="path to input dir"
    )
    args = parser.parse_args()
    filenames = glob(f"{args.in_dir}/**/*.wav", recursive=True)+glob(f"{args.in_dir}/**/*.flac", recursive=True)  # [:10]
    in_dir = args.in_dir
    if in_dir[-1]=='\/':
        in_dir=in_dir[:-1]
    process_batch(filenames)
    # num_threads = 4
    # funcs = []
    # for i in range(num_threads):
    #     funcs.append(functools.partial(process_batch, filenames[i::num_threads]))
    # with multiprocessing.Pool() as pool:
    #     results = [pool.apply_async(func) for func in funcs]
    #     pool.close()
    #     pool.join()
    # for result in results:
    #     result.get()
