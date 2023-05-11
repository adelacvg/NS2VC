import io
import logging
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='ns2vc inference')

    # Required
    parser.add_argument('-m', '--model_path', type=str, default="logs/model-10.pt",
                        help='Path to the model.')
    parser.add_argument('-c', '--config_path', type=str, default="config.json",
                        help='Path to the configuration file.')
    parser.add_argument('-r', '--refer_names', type=str, default=["1.wav"],
                        help='Reference audio path.')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["2.wav"],
                        help='A list of wav file names located in the raw folder.')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0],
                        help='Pitch adjustment, supports positive and negative (semitone) values.')

    # Optional
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False,
                        help='Automatic pitch prediction for voice conversion. Do not enable this when converting songs as it can cause serious pitch issues.')
    parser.add_argument('-cl', '--clip', type=float, default=0,
                        help='Voice forced slicing. Set to 0 to turn off(default), duration in seconds.')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0,
                        help='The cross fade length of two audio slices in seconds. If there is a discontinuous voice after forced slicing, you can adjust this value. Otherwise, it is recommended to use. Default 0.')
    parser.add_argument('-fmp', '--f0_mean_pooling', action='store_true', default=False,
                        help='Apply mean filter (pooling) to f0, which may improve some hoarse sounds. Enabling this option will reduce inference speed.')

    # generally keep default
    parser.add_argument('-sd', '--slice_db', type=int, default=-40,
                        help='Loudness for automatic slicing. For noisy audio it can be set to -30')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device used for inference. None means auto selecting.')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5,
                        help='Due to unknown reasons, there may be abnormal noise at the beginning and end. It will disappear after padding a short silent segment.')
    parser.add_argument('-wf', '--wav_format', type=str, default='wav',
                        help='output format')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75,
                        help='Proportion of cross length retention, range (0-1]. After forced slicing, the beginning and end of each segment need to be discarded.')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,
                        help='F0 Filtering threshold: This parameter is valid only when f0_mean_pooling is enabled. Values range from 0 to 1. Reducing this value reduces the probability of being out of tune, but increases matte.')


    args = parser.parse_args()

    clean_names = args.clean_names
    refer_names = args.refer_names
    trans = args.trans
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    F0_mean_pooling = args.f0_mean_pooling
    cr_threshold = args.f0_filter_threshold

    svc_model = Svc(args.model_path, args.config_path, args.device)
    raw_folder = "dataset"
    results_folder = "output"
    infer_tool.mkdir([raw_folder, results_folder])

    infer_tool.fill_a_to_b(trans, clean_names)
    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"{raw_folder}/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        per_size = int(clip*audio_sr)
        lg_size = int(lg*audio_sr)
        lg_size_r = int(lg_size*lgr)
        lg_size_c_l = (lg_size-lg_size_r)//2
        lg_size_c_r = lg_size-lg_size_r-lg_size_c_l
        lg = np.linspace(0,1,lg_size_r) if lg_size!=0 else 0

        for refer_name in refer_names:
            audio = []
            refer_path = f"{raw_folder}/{refer_name}"
            if "." not in refer_path:
                refer_path += ".wav"
            infer_tool.format_wav(refer_path)
            refer_path = Path(refer_path).with_suffix('.wav')
            for (slice_tag, data) in audio_data:
                print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
                
                length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
                if slice_tag:
                    print('jump empty segment')
                    _audio = np.zeros(length)
                    audio.extend(list(infer_tool.pad_array(_audio, length)))
                    continue
                if per_size != 0:
                    datas = infer_tool.split_list_by_n(data, per_size,lg_size)
                else:
                    datas = [data]
                # print(len(datas))
                for k,dat in enumerate(datas):
                    per_length = int(np.ceil(len(dat) / audio_sr * svc_model.target_sample)) if clip!=0 else length
                    if clip!=0: print(f'###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                    # padd
                    pad_len = int(audio_sr * pad_seconds)
                    dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                    raw_path = io.BytesIO()
                    soundfile.write(raw_path, dat, audio_sr, format="wav")
                    raw_path.seek(0)
                    out_audio, out_sr = svc_model.infer(tran, raw_path, refer_path,
                                                        auto_predict_f0=auto_predict_f0,
                                                        F0_mean_pooling = F0_mean_pooling,
                                                        cr_threshold = cr_threshold
                                                        )
                    # print(1)
                    # print(out_audio.shape)
                    _audio = out_audio.cpu().numpy()
                    pad_len = int(svc_model.target_sample * pad_seconds)
                    _audio = _audio[pad_len:-pad_len]
                    _audio = infer_tool.pad_array(_audio, per_length)
                    if lg_size!=0 and k!=0:
                        lg1 = audio[-(lg_size_r+lg_size_c_r):-lg_size_c_r] if lgr != 1 else audio[-lg_size:]
                        lg2 = _audio[lg_size_c_l:lg_size_c_l+lg_size_r]  if lgr != 1 else _audio[0:lg_size]
                        lg_pre = lg1*(1-lg)+lg2*lg
                        audio = audio[0:-(lg_size_r+lg_size_c_r)] if lgr != 1 else audio[0:-lg_size]
                        audio.extend(lg_pre)
                        _audio = _audio[lg_size_c_l+lg_size_r:] if lgr != 1 else _audio[lg_size:]
                    audio.extend(list(_audio))
                    # print(1)
            key = "auto" if auto_predict_f0 else f"{tran}key"
            res_path = f'./{results_folder}/{clean_name}_{key}_{refer_name}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            
if __name__ == '__main__':
    main()
