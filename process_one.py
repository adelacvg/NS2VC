import torchaudio
import torchaudio.transforms as T
import torch
import utils
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
hmodel = utils.get_hubert_model().to(device)
def process_one(in_dir, filename):
    wav, sr = torchaudio.load(filename)
    if wav.shape[0] > 1:  # mix to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav16k = T.Resample(sr, 16000)(wav)
    wav24k = T.Resample(sr, 24000)(wav)
    filename = filename.replace(in_dir, in_dir+"_processed").replace('.mp3','.wav').replace('.flac','.wav')
    wav24k_path = filename
    if not os.path.exists(os.path.dirname(wav24k_path)):
        os.makedirs(os.path.dirname(wav24k_path))
    torchaudio.save(wav24k_path, wav24k, 24000)
    cvec_path = filename + ".cvec.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav16k = wav16k.to(device)
    c = utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k[0])
    torch.save(c.cpu(), cvec_path)

    spec_path = filename.replace(".wav", ".mel.pt")
    spec_process = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        center=True,
        power=1,
    )
    spec = spec_process(wav24k)# 1 100 T
    spec = torch.log(torch.clip(spec, min=1e-7))
    torch.save(spec, spec_path)

