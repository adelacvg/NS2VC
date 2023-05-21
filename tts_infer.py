import json
import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from encodec_wrapper import EncodecWrapper
from g2p_en import G2p
from model import NaturalSpeech2
from pypinyin import pinyin, Style

from ema_pytorch import EMA
from dataset import TextDataset
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
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

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
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
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


def synthesize(model, cfg, encodec, batchs, control_values, device):
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        phoneme, refer_path, phoneme_length = batch 
        phoneme = phoneme.to(device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


def load_model(model_path, device, cfg):
    data = torch.load(model_path, map_location=device)
    model = NaturalSpeech2(cfg=cfg)
    model.load_state_dict(data['model'])

    return model.eval()
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        default="single",
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
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
        required=True,
        help="language of the input text",
    )
    parser.add_argument(
        "--refer",
        type=str,
        default="dataset/1.wav",
        help="reference audio path for single-sentence mode only",
    )
    parser.add_argument(
        "-c", "--config_path", type=str, default="config.json", required=True, help="path to config.json"
    )
    parser.add_argument(
        "-m", "--model_path", type=str, default="logs/model-3.pt", required=True, help="path to model.pt"
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
    if args.mode == "single":
        assert args.text is not None

    # Read Config

    cfg = json.load(open(args.config_path))

    # Get model
    model = load_model(args.model_path, device, cfg)

    # Load vocoder
    encodec = EncodecWrapper().to(device)

    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        if args.lang == "en":
            texts = np.array([preprocess_english(args.text, cfg)])
        elif args.lang == "zh":
            texts = np.array([preprocess_mandarin(args.text, cfg)])
        text_lens = np.array([len(texts[0])])
        refer_path = args.refer
        batchs = [( texts,refer_path,text_lens)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, cfg, encodec, batchs, control_values, device)

