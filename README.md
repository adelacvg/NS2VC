
# NaturalSpeech2

# This branch is for the tts task, the vc task is in the master branch.

## Unofficial implementation of <a href="https://arxiv.org/pdf/2304.09116.pdf">NaturalSpeech2</a> for Voice Conversion

### Data preprocessing

You need to use mfa to align the data first.
In this project, I use the`mfa align ./dataset english_us_arpa english_us_arpa ./dataset` command to align the data.

Put the data in the dataset folder, and then run the following command to preprocess the data.

```python
python preprocess.py
```

### Training
Run `accelerate config` to generate the config file, and then train the model.

```python
```python
accelerate launch train.py
```

### Q&A

qq group:801645314
You can add the qq group to discuss the project.

Thanks to <a href="https://github.com/svc-develop-team/so-vits-svc/">sovits4</a>, <a href="https://github.com/lucidrains/naturalspeech2-pytorch/">naturalspeech2</a> and <a href="https://github.com/lucidrains/imagen-pytorch">imagen</a> for their great works.
