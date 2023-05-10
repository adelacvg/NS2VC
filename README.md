
# NaturalSPeech2VC

## Unofficial implementation of <a href="https://arxiv.org/pdf/2304.09116.pdf">NaturalSpeech2</a> for Voice Conversion

### Requirements
You can install the most of requirements by running the following command.
```python
pip install audiolm_pytorch
```
Put the pretrained contentvec model under hubert folder.
You can download the pretrained contentvec model from <a href="https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr">here</a>.

### Data preprocessing
Put the wav files under dataset folder.
```
dataset
├───speaker0
|   ├───xxx
│   │   ├───xxx-xxx.wav
|   │   └───xxx-xxx.wav
│   ├───xxx-yyy.wav
│   ├───...
│   └───xxx-yyy.wav
└───xxx.wav
```
Run the following command to preprocess the data.
```python
python preprocess.py
```

### Training

Install the accelerate first by `pip install accelerate`, run `accelerate config` to configure the environment, and then run the following command to train the model.

```python
accelerate launch train.py
```

### Q&A

qq group:801645314

You can add the qq group to discuss the project.

Thanks to <a href="https://github.com/svc-develop-team/so-vits-svc/">sovits4</a> and <a href="https://github.com/lucidrains/naturalspeech2-pytorch/">naturalspeech2</a>.
