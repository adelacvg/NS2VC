
# NaturalSPeech2VC(WIP)

## Unofficial implementation of <a href="https://arxiv.org/pdf/2304.09116.pdf">NaturalSpeech2</a> for Voice Conversion

### Data preprocessing

Put the data in the dataset folder, and then run the following command to preprocess the data.

```python
python preprocess.py
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

### TTS
If you want to use the TTS model, please check the TTS branch.

### Q&A

qq group:801645314
You can add the qq group to discuss the project.

Thanks to <a href="https://github.com/svc-develop-team/so-vits-svc/">sovits4</a>, <a href="https://github.com/lucidrains/naturalspeech2-pytorch/">naturalspeech2</a> and <a href="https://github.com/lucidrains/imagen-pytorch">imagen</a> for their great works.
