
# NaturalSpeech2_v2

# This branch is for the tts task, the vc task is in the vc_v2 branch.

## Unofficial implementation of <a href="https://arxiv.org/pdf/2304.09116.pdf">NaturalSpeech2</a> for TTS
### Dataset
You should put your dataset in the dataset folder, and the dataset should be organized as follows:

```
dataset
├── p225
│   ├── p225_001.wav
|   ├── p225_001.txt
|   ├── P225_001.TextGrid
│   ├── p225_002.wav
|   ├── p225_002.txt
|   ├── P225_002.TextGrid
│   ├── ...
├── p226
│   ├── p226_001.wav
|   ├── p226_001.txt
|   ├── P226_001.TextGrid
│   ├── p226_002.wav
|   ├── p226_002.txt
|   ├── P226_002.TextGrid
│   ├── ...
```
and processed dataset will be saved in the folder with _processed suffix under the same father folder as the dataset folder.

### Data preprocessing

You need to use mfa to align the data first.
In this project, I use the`mfa align ./dataset english_us_arpa english_us_arpa ./dataset` command to align the data.

For the mandarin alignment, you can use`mfa align dataset/ mandarin_mfa_tools/simple.txt mandarin_mfa_tools/aishell3_model.zip dataset/` command to align the data.

Put the textgird files in the dataset folder, and then run the following command to preprocess the data.

```python
python preprocess.py
```

### Requirements
Use the following command to initialize the env and install the requirements.
```bash
bash init.sh
```
And env named vocos will be created.

### Training
Run `accelerate config` to generate the config file, and then train the model.

```python
accelerate launch train.py
```
### Inference
Run `python tts_infer.py` for inference. You should change the text and model_path in the tts_infer.py file before running the command.
And set the language to `zh` or `en` which you would like to use.

### Memory Cost
For training the vc model, with batch size of 8, the memory cost is about 10G.
For training the tts model, with batch size of 8, the memory cost is about 14G.

### About pretrained model
May be coming soon.

### Q&A
qq group:801645314
You can add the qq group to discuss the project.

Thanks to <a href="https://github.com/svc-develop-team/so-vits-svc/">sovits4</a>, <a href="https://github.com/lucidrains/naturalspeech2-pytorch/">naturalspeech2</a> and <a href="https://github.com/lucidrains/imagen-pytorch">imagen</a> for their great works.
