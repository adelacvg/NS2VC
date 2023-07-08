
# NaturalSpeech2

# This branch is for the tts task, the vc task is in the master branch.

## Unofficial implementation of <a href="https://arxiv.org/pdf/2304.09116.pdf">NaturalSpeech2</a> for Voice Conversion
### Dataset
You should put your dataset in the dataset folder, and the dataset should be organized as follows:

```
dataset
├── p225
│   ├── p225_001.wav
|   ├── P225_001.TextGrid
│   ├── p225_002.wav
|   ├── P225_002.TextGrid
│   ├── ...
├── p226
│   ├── p226_001.wav
|   ├── P226_001.TextGrid
│   ├── p226_002.wav
|   ├── P226_002.TextGrid
│   ├── ...
```
and processed dataset will be saved in the processed_dataset folder under the same folder as the dataset folder.

### Data preprocessing

You need to use mfa to align the data first.
In this project, I use the`mfa align ./dataset english_us_arpa english_us_arpa ./dataset` command to align the data.

For the mandarin alignment, you can use`mfa align dataset/ mandarin_mfa_tools/simple.txt mandarin_mfa_tools/aishell3_model.zip aligned_dataset` command to align the data.

Put the textgird files in the dataset folder, and then run the following command to preprocess the data.

```python
python preprocess.py
```

### Requirements
You can install the most requirements by running the following command.

```python
pip install audiolm
```

### Training
Run `accelerate config` to generate the config file, and then train the model.

For the mandarin training, you should change the language in config file from `en` to `zh`.

```python

```python
accelerate launch train.py
```
### Inference
Run `python tts_infer.py` for inference. You should change the text and model_path in the tts_infer.py file before running the command.

### Memory Cost
For training the vc model, with batch size of 8, the memory cost is about 18G.
For training the tts model, with batch size of 8, the memory cost is about 13G.

### About pretrained model
The pretrained model is trained on `cc58c2d` commit and is not compatible with later code.

### Q&A
qq group:801645314
You can add the qq group to discuss the project.

Thanks to <a href="https://github.com/svc-develop-team/so-vits-svc/">sovits4</a>, <a href="https://github.com/lucidrains/naturalspeech2-pytorch/">naturalspeech2</a> and <a href="https://github.com/lucidrains/imagen-pytorch">imagen</a> for their great works.
