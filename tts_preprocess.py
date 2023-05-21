import os
wav_folder = 'vctk/VCTK-Corpus/wav48'
txt_folder = 'vctk/VCTK-Corpus/txt'

for spk in os.listdir(txt_folder):
    #copy txt to wav
    for txt in os.listdir(os.path.join(txt_folder, spk)):
        txt_path = os.path.join(txt_folder, spk, txt)
        wav_path = os.path.join(wav_folder, spk)
        os.system('cp {} {}'.format(txt_path, wav_path))
