import math
import multiprocessing
import os
import argparse
from random import shuffle
import torchaudio
import torchaudio.transforms as T

import torch
from glob import glob
from tqdm import tqdm

from audiolm_pytorch import SoundStream, EncodecWrapper
import utils
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa
import numpy as np
import tgt

hps = utils.get_hparams_from_file("config.json")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length

def process_one(filename, hmodel, codec):
    wav, sr = torchaudio.load(filename)
    wav16k = T.Resample(sr, 16000)(wav)
    wav24k = T.Resample(sr, 24000)(wav)
    # wav24k_path = filename + ".24k.wav"
    # torchaudio.save(wav24k_path, wav24k, 24000)
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav16k = wav16k.to(device)
        c = utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k[0])
        torch.save(c.cpu(), soft_path)

    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0 = utils.compute_f0_dio(
            wav24k.cpu().numpy()[0], sampling_rate=24000, hop_length=320
        )
        np.save(f0_path, f0)

    codes_path = filename.replace(".wav", ".code.pt")
    if not os.path.exists(codes_path):
        wav24k = wav24k.unsqueeze(0)
        codec.eval()
        codes, _, _ = codec(wav24k, return_encoded = True)
        codes = torch.squeeze(codes, 0).transpose(1,2)
        # print(codes.shape)#24k 1 128 T2+1
        torch.save(codes, codes_path)


def process_batch(filenames):
    print("Loading hubert for content...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmodel = utils.get_hubert_model().to(device)
    codec = EncodecWrapper()
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename, hmodel, codec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset", help="path to input dir"
    )

    args = parser.parse_args()
    filenames = glob(f"{args.in_dir}/**/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    process_batch(filenames)