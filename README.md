
# NS2VC_v2

## Unofficial implementation of <a href="https://arxiv.org/pdf/2304.09116.pdf">NaturalSpeech2</a> for Voice Conversion
Different from the NS2, I use the vocos but encodec as the vocoder for better quality, and use contentvec to substitute the text embedding and duration span process. 
I also adopted the unet1d conditional model from the diffusers lib, thanks for their hard works.

### About Zero shot generalization
I did many attempt on improve the generalization of the model. And I find that it's much like the stable diffusion. If a tag is not in your train set, you can't get a promising result, so larger dataset, more speaker, better generalization, better results. The model can ensure speaker in trainset have a good result.
### Demo
| refer | input | output |
| :----| :---- | :---- |
| [refer0.webm](https://github.com/adelacvg/NS2VC/assets/27419496/abed2fdc-8366-4522-bbc7-646e0ae6b842)| [gt0.webm](https://github.com/adelacvg/NS2VC/assets/27419496/327794b0-e550-4932-8075-4be09e063d45)| [gen0.webm](https://github.com/adelacvg/NS2VC/assets/27419496/3defcd4a-6843-464c-a903-285a14751096)|
| [refer1.webm](https://github.com/adelacvg/NS2VC/assets/27419496/3d924019-0a68-41a5-aeaf-928a9b8fa8b5)| [gt1.webm](https://github.com/adelacvg/NS2VC/assets/27419496/12fc1514-0edb-493d-a07f-3c94b0548557)| [gen1.webm](https://github.com/adelacvg/NS2VC/assets/27419496/f38e8780-1baf-48b5-b6e5-0ba3856599e2)|
|[refer2.webm](https://github.com/adelacvg/NS2VC/assets/27419496/9759088b-10e7-4bb1-a0ed-c808e11b9f9e)|[gt2.webm](https://github.com/adelacvg/NS2VC/assets/27419496/ddff8bfc-7c6a-4d53-9b98-0d66c421d1d1)|[gen2.webm](https://github.com/adelacvg/NS2VC/assets/27419496/d72cb17d-6813-4d87-8ec5-929b2cc2fb15)|
|[refer3.webm](https://github.com/adelacvg/NS2VC/assets/27419496/c9e045ac-914c-4b49-a112-c71acce2eb27)|[gt3.webm](https://github.com/adelacvg/NS2VC/assets/27419496/a684e11d-32fe-46e3-87e0-e0c6047a24dc)|[gen3.webm](https://github.com/adelacvg/NS2VC/assets/27419496/df3ceced-bfae-4272-a8d7-94a49826f04a)|
|[refer4.webm](https://github.com/adelacvg/NS2VC/assets/27419496/e3191a18-44fc-477e-9ed4-60c42ad35b80)|[gt4.webm](https://github.com/adelacvg/NS2VC/assets/27419496/318a0843-89a5-46de-b1e2-2039a457bc17)|[gen4.webm](https://github.com/adelacvg/NS2VC/assets/27419496/06487dab-f047-4461-9e5c-4bd53bfdfd56)|




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
### Continue training
If you want to fine tune or continue to train a model.
Add
```python
trainer.load('your_model_path')
```
to the `train.py`.
### Pretrained model
Maybe comming soon, if I had enough data for a good model.

### TTS

If you want to use the TTS model, please check the TTS branch.

### Q&A

qq group:801645314
You can add the qq group to discuss the project.

Thanks to <a href="https://github.com/svc-develop-team/so-vits-svc/">sovits4</a>, <a href="https://github.com/lucidrains/naturalspeech2-pytorch/">naturalspeech2</a> and <a href="https://github.com/lucidrains/imagen-pytorch">imagen</a> <a href="https://github.com/huggingface/diffusers">diffusers</a>for their great works.
