
# NaturalSPeech2VC(WIP)

## Unofficial implementation of <a href="https://arxiv.org/pdf/2304.09116.pdf">NaturalSpeech2</a> for Voice Conversion

### Data preprocessing
First of all, you need to download the contentvec model and put it under the hubert folder.
The model can be download from <a href="https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr">here</a>.

The dataset structure can be like this:

```
dataset
├── spk1
│   ├── 1.wav
│   ├── 2.wav
│   ├── ...
│   └── spk11
│       ├── 11.wav
├── 3.wav
├── 4.wav
```

Overall, you can put the data in any way you like.

Put the data with .wav extension under the dataset folder, and then run the following command to preprocess the data.

```python
python preprocess.py
```

The preprocessed data will be saved under the processed_dataset folder.

## Requirements

You can install the most of the requirements by running the following command.

```python
pip install audiolm
```

### Training

Install the accelerate first, run `accelerate config` to configure the environment, and then run the following command to train the model.

```python
accelerate launch train.py
```

### Inference

Change the device, model_path, clean_names and refer_names in the inference.py, and then run the following command to inference the model.

```python
python infer.py
```

### Pretrained model
Download the pretrained tts or vc model from <a href="https://huggingface.co/adelacvg/NS2VC">here</a>.

### TTS

If you want to use the TTS model, please check the TTS branch.

### Q&A

qq group:801645314
You can add the qq group to discuss the project.

Thanks to <a href="https://github.com/svc-develop-team/so-vits-svc/">sovits4</a>, <a href="https://github.com/lucidrains/naturalspeech2-pytorch/">naturalspeech2</a> and <a href="https://github.com/lucidrains/imagen-pytorch">imagen</a> for their great works.
