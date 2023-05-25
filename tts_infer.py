import json
import re
import argparse
from string import punctuation

import torch
import torchaudio
import torchaudio.transforms as T
import yaml
import numpy as np
from torch.utils.data import DataLoader
from encodec_wrapper import EncodecWrapper
from g2p_en import G2p
from model import NaturalSpeech2
from pypinyin import pinyin, Style

from ema_pytorch import EMA
from text import text_to_sequence


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon('./lexicons/librispeech-lexicon.txt')

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    cleaners = ["english_cleaners"]
    sequence = np.array(
        text_to_sequence(
            phones, cleaners
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, cfg, codec, batchs, control_values, device):
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        phoneme, refer_path, phoneme_length = batch 
        phoneme = torch.LongTensor(phoneme).to(device)
        phoneme_length = torch.LongTensor(phoneme_length).to(device)
        refer_audio,sr = torchaudio.load(refer_path)
        refer_audio24k = T.Resample(sr, 24000)(refer_audio).to(device)
        codes, _, _ = codec(refer_audio24k, return_encoded = True)
        refer = codes.transpose(1,2)
        refer_length = torch.tensor([refer.size(1)]).to(device)
        with torch.no_grad():
            samples = model.sample(phoneme, refer, phoneme_length, refer_length, codec).detach().cpu()
    return samples
def load_model(model_path, device, cfg):
    data = torch.load(model_path, map_location=device)
    model = NaturalSpeech2(cfg=cfg)
    model.load_state_dict(data['model'])

    ema = EMA(model)
    ema.to(device)
    ema.load_state_dict(data["ema"])
    return ema.ema_model.eval()
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="Please call Stella.",
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "zh"],
        default="en",
        help="language of the input text",
    )
    parser.add_argument(
        "--refer",
        type=str,
        default="1.wav",
        help="reference audio path for single-sentence mode only",
    )
    parser.add_argument(
        "-c", "--config_path", type=str, default="config.json", help="path to config.json"
    )
    parser.add_argument(
        "-m", "--model_path", type=str, default="logs/model-34.pt", help="path to model.pt"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    device = "cuda:1"
    # Check source texts
    assert args.text is not None

    # Read Config

    cfg = json.load(open(args.config_path))

    # Get model
    model = load_model(args.model_path, device, cfg)

    # Load vocoder
    codec = EncodecWrapper().eval().to(device)

    ids = raw_texts = [args.text[:100]]
    if args.lang == "en":
        texts = np.array([preprocess_english(args.text, cfg)])
    elif args.lang == "zh":
        texts = np.array([preprocess_mandarin(args.text, cfg)])
    text_lens = np.array([len(texts[0])])
    raw_path = 'raw'
    refer_name = args.refer
    refer_path = f"{raw_path}/{refer_name}"
    batchs = [( texts,refer_path,text_lens)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    audios = synthesize(model, cfg, codec, batchs, control_values, device)

    results_folder = "output"
    result_path = f'./{results_folder}/tts_{refer_name}.wav'
    torchaudio.save(result_path, audios, 24000)
