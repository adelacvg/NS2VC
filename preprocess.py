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
in_dir = ""
def get_alignment(tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # print(1,p)
            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s
            if p not in sil_phones:
                # For ordinary phones
                # print(p)
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                if p=="":
                    p='sil'
                phones.append(p)

            durations.append(
                int(
                    np.round(e * sampling_rate / hop_length)
                    - np.round(s * sampling_rate / hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time


def process_one(filename, hmodel, codec):
    # print(filename)
    textgrid_path = filename.replace(".wav", ".TextGrid")
    try:
        textgrid = tgt.io.read_textgrid(textgrid_path)
    except:
        print("Error reading textgrid:", textgrid_path)
        return
    phone, duration, start, end = get_alignment(
        textgrid.get_tier_by_name("phones")
    )
    text = "{" + " ".join(phone) + "}"
    wav, sr = torchaudio.load(filename)
    wav = wav[:,int(sr * start) : int(sr * end)]
    wav = torch.mean(wav, dim=0).unsqueeze(0)
    wav16k = T.Resample(sr, 16000)(wav)
    wav24k = T.Resample(sr, 24000)(wav)
    filename = filename.replace(in_dir, in_dir+"_processed")
    wav24k_path = filename
    if not os.path.exists(os.path.dirname(wav24k_path)):
        os.makedirs(os.path.dirname(wav24k_path))
    torchaudio.save(wav24k_path, wav24k, 24000)
    duration_path = filename + ".duration.npy"
    np.save(duration_path, np.array(duration))

    text_path = filename+".phone.txt"
    with open(text_path, 'w') as f:
        f.write(text)

    # soft_path = filename + ".soft.pt"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # wav16k = wav16k.to(device)
    # c = utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k[0])
    # torch.save(c.cpu(), soft_path)

    f0_path = filename + ".f0.npy"

    f0 = utils.compute_f0_dio(
        wav24k.cpu().numpy()[0], sampling_rate=24000, hop_length=320
    )
    # print(f0.shape)#24k  T2 
    np.save(f0_path, f0)

    codes_path = filename.replace(".wav", ".code.pt")

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
    in_dir = args.in_dir
    shuffle(filenames)
    process_batch(filenames)
