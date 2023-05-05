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

hps = utils.get_hparams_from_file("config.json")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length


def process_one(filename, hmodel, codec):
    # print(filename)
    wav, sr = librosa.load(filename, sr=sampling_rate)
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k)
        # print(c.shape)#16k  1 256 T1
        torch.save(c.cpu(), soft_path)

    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0 = utils.compute_f0_dio(
            wav, sampling_rate=sampling_rate, hop_length=hop_length
        )
        # print(f0.shape)#24k  T2 
        np.save(f0_path, f0)

    codes_path = filename.replace(".wav", ".code.pt")
    if not os.path.exists(codes_path):
        # Process spectrogram
        # The following code can't be replaced by torch.FloatTensor(wav)
        # because load_wav_to_torch return a tensor that need to be normalized
        audio, sr = torchaudio.load(filename)
        audio24k = T.Resample(sr, 24000)(audio)

        audio24k = audio24k.unsqueeze(0)

        # audio_path = filename.replace(".wav", "_24k.wav")
        # torchaudio.save(audio_path, audio24k, 24000)

        codec.eval()
        codes, _, _ = codec(audio24k, return_encoded = True)
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
    filenames = glob(f"{args.in_dir}/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks
    ]
    for p in processes:
        p.start()