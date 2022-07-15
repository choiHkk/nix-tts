## Introduction
### Nix-TTS unofficial pytorch implementation
1. nix-tts를 구현하여 vits 오픈 소스와 함께 한국어 데이터셋(KSS)을 사용해 빠르게 학습합니다.
2. KSS 데이터셋은 기본적으로 44kHz인 점을 감안하여 22kHz로 resampling할 수 있도록 utils.load_wav_to_torch()를 수정했습니다.
3. data_utils.py 내부 get_audio method에는 spectrogram을 생성하고 저장한 뒤 다시 읽어오게끔 하는 부분이 있지만 해당 부분을 주석처리했습니다.
4. data_utils.py 내부 get_text method를 kss 스크립트에 맞추어 변경했습니다.
5. utils.py 내부 load_filepaths_and_text func. 를 data_utils.py 입력 형식에 맞게끔 수정했습니다.
6. stft_loss.py 를 추가하고 models.py SynthesizerTrn 부분 speaker embedding if문 일부를 수정했습니다.
7. conda 환경으로 진행해도 무방하지만 본 레포지토리에서는 docker 환경만 제공합니다. 기본적으로 ubuntu에 docker, nvidia-docker가 설치되었다고 가정합니다.
8. GPU, CUDA 종류에 따라 Dockerfile 상단 torch image 수정이 필요할 수도 있습니다.
9. 오픈소스에서 제공하는 영어 문장은 국제음성기호(ipa)를 사용하지만 본 레포지토리에서는 일반적인 한국어 전처리 기법을 사용합니다.
10. 별도의 pre-processing 과정은 필요하지 않습니다.


## Dataset
1. download dataset - https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
2. `unzip /path/to/the/kss.zip -d /path/to/the/kss`
3. `mkdir /path/to/the/vits/data/dataset`
4. `mv /path/to/the/kss.zip /path/to/the/nix-tts/data/dataset`

## Docker build
1. `cd /path/to/the/nix-tts`
2. `docker build --tag nix-tts:latest .`

## Training
1. `nvidia-docker run -it --name 'nix-tts' -v /path/to/nix-tts:/home/work/nix-tts --ipc=host --privileged nix-tts:latest`
2. `cd /home/work/nix-tts/monotonic_align`
3. `python setup.py build_ext --inplace`
4. `cd /home/work/nix-tts`
5. `git clone https://github.com/jaywalnut310/vits`
6. `cd /home/work/nix-tts/vits/monotonic_align`
7. `python setup.py build_ext --inplace`
8. `cd /home/work/nix-tts`
9. `ln -s /path/to/the/nix-tts/data/dataset/kss`
10. `python train_encoder.py -c ./config/kss_distil_encoder.json -m kss -g 0`
11. `python train_decoder.py -c ./config/kss_distil_decoder.json -m kss -g 0`
12. arguments
  * -c : comfig path
  * -m : model output directory
  * -g : gpu number
13. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Tensorboard losses
![nix-tts-tensorboard-losses](https://user-images.githubusercontent.com/69423543/179240643-d0be3733-c19a-4f33-ae4a-1fa255ddd191.png)

## Tensorboard alignment
![nix-tts-tensorboard-alignment](https://user-images.githubusercontent.com/69423543/179240657-8090b2f0-1e16-43c6-9167-7e88141770e3.png)

## Tensorboard mel-spectrograms
![nix-tts-tensorboard-mels](https://user-images.githubusercontent.com/69423543/179240889-0d39f2a7-309a-4741-81fd-aa0f91203cc5.png)


## Reference
1. [Nix-TTS: An Incredibly Lightweight End-to-End Text-to-Speech Model via Non End-to-End Distillation](https://arxiv.org/abs/2203.15643)
2. [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103)
3. [SpeedySpeech: Efficient Neural Speech Synthesis](https://arxiv.org/abs/2008.03802)
4. [RAD-TTS: Parallel Flow-Based TTS with Robust Alignment Learning and Diverse Synthesis](https://openreview.net/pdf?id=0NQwnnwAORi)
5. [One TTS Alignment To Rule Them All](https://arxiv.org/pdf/2108.10447.pdf)
