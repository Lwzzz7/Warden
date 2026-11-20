# Warden: Towards Practical Generative Speech Watermarking via Parameter-Efficient Injection

**Weizhi Liu, Yue Li, and Zhaoxia Yin**

[Paper (arXiv)](https://arxiv.org/abs/2601.22556) · [Listen to the samples](https://lwzzz7.github.io/Warden_Demo/)

Warden embeds a source identifier while a neural vocoder synthesizes speech. A lightweight **Temporal Adapter** aligns the watermark bits with intermediate acoustic features and distributes them across frames. A **Coarse-to-Fine Gated Extractor** reads the bits back from the waveform. Training uses extraction accuracy to adjust the balance between watermark recovery and audio fidelity while keeping the pretrained vocoder frozen.

On LJSpeech, Warden recovered **96.22% of watermark bits** on average across four vocoder configurations and eight neural codec regeneration settings at a payload of 100 bits per second. The adapter and extractor together have **1.66M trainable parameters** for the diffusion vocoders and **1.76M** for the GAN vocoders. The audio demo lets you compare original and watermarked speech for the same input.

## Installation

Run the examples from the repository root. The reference environment uses Python 3.8, PyTorch 1.10.1, and torchaudio 0.10.1. The commands below select the CPU build. For GPU training, install the matching CUDA build in the same environment. [PyTorch installation options](https://pytorch.org/get-started/previous-versions/#v1101)

```bash
conda create -n warden python=3.8
conda activate warden
conda install pytorch==1.10.1 torchaudio==0.10.1 cpuonly -c pytorch
python -m pip install -r requirements.txt
```

## Prepare the models and watermark

Choose one vocoder. Download its pretrained generator and place it at the path in the table, relative to the repository root. The HiFi-GAN and BigVGAN configurations are included.

| Vocoder | `--model` | Commands | Pretrained generator | Expected path |
| --- | --- | --- | --- | --- |
| DiffWave | `DiffWave` | Train, test | [22.05 kHz checkpoint](https://github.com/lmnt-com/diffwave#pretrained-models) | `diffwave/pretrain/diffwave-22kHz.pt` |
| HiFi-GAN | `HifiGAN` | Train, test | [LJ_V1 generator](https://github.com/jik876/hifi-gan#pretrained-model) | `hifigan/pretrain/generator_v1` |
| BigVGAN | `BigvGAN` | Train, test | [22 kHz, 80-band base generator](https://huggingface.co/nvidia/bigvgan_base_22khz_80band/blob/main/bigvgan_generator.pt) | `bigvgan/pretrain/bigvgan_generator.pt` |
| PriorGrad | `PriorGrad` | Test only | Uses the DiffWave backbone, adapter checkpoint, and sampler | `diffwave/pretrain/diffwave-22kHz.pt` |

Copy the included 100-bit watermark pool to the directory used by both scripts. Each row contains a binary identifier. The trainable adapter and extractor weights are saved separately by `train.py`.

```bash
mkdir -p watermarks
cp watermark100.pt watermarks/watermark100.pt
```

The examples use `--assets-root .`. If you store the vocoder assets elsewhere, keep the same relative paths under that directory, including [the HiFi-GAN config](hifigan/config_v1.json) or [the BigVGAN config](bigvgan/configs/bigvgan_base_22khz_80band.json) when applicable.

## Prepare the data

Download and extract [LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/). The example below copies the files in the [training](splits/LJSpeech_train.txt) and [test](splits/LJSpeech_test.txt) lists into separate directories. Set the source path to the extracted `wavs/` folder.

```bash
wav_dir=/path/to/LJSpeech-1.1/wavs
for split in train test; do
  mkdir -p "data/LJSpeech/$split"
  rsync -a --files-from="splits/LJSpeech_${split}.txt" "$wav_dir/" "data/LJSpeech/$split/"
done
```

The LJSpeech loader reads mono 22,050 Hz WAVs directly from `--data-root`. It takes the first second of each recording and zero-pads shorter clips.

For [LibriTTS](https://www.openslr.org/60/), use `--dataset LibriTTS` and pass the parent of the extracted `LibriTTS/` directory. Evaluation uses `test-clean`. For [AISHELL-3](https://www.openslr.org/93/), use `--dataset Aishell` and point to the desired split's WAV directory. The loader searches it recursively. Both loaders resample audio to 22,050 Hz.

## Train

The following command trains a 100-bit adapter and extractor for DiffWave. Training samples a distortion before extraction and loads the 24 kHz EnCodec model, so allow its weights to download on first use or populate the PyTorch cache beforehand.

```bash
python train.py --model DiffWave --dataset LJSpeech \
  --data-root data/LJSpeech/train --assets-root . \
  --watermark-length 100 --output checkpoints/diffwave
```

The defaults are 50 epochs, batch size 4, and an AdamW learning rate of `5e-4`. Training writes `last.pt` at completion, a checkpoint every five epochs, and `best.pt` when the average training loss improves after epoch three. For a one-batch setup check, add `--epochs 1 --max-steps 1` and choose a separate output directory. Use `--resume PATH` to initialize the adapter and extractor from an existing Warden checkpoint with a fresh optimizer.

## Evaluate watermark recovery

Use the checkpoint produced above, or supply a compatible checkpoint containing both the adapter and extractor for the selected vocoder and payload. The included `watermark100.pt` supplies the reference bits. Run a clean evaluation first:

```bash
python test.py --model DiffWave --dataset LJSpeech \
  --data-root data/LJSpeech/test --assets-root . \
  --watermark-length 100 --checkpoint checkpoints/diffwave/last.pt \
  --output results/diffwave-clean.json
```

The script synthesizes watermarked speech from each input, extracts the identifier, and writes per-clip results and their means to JSON. **`mean_bit_accuracy`** measures the fraction of reference bits recovered correctly. **`mean_stft_loss`** and **`mean_mel_loss`** compare the generated waveform with the source recording. Add `--threshold T` with a separately calibrated bit-agreement threshold to report true- and false-positive rates, using original recordings as negatives.

To evaluate **EnCodec regeneration**, add `--attack 4` and change the output to `results/diffwave-encodec.json`. The default `clean` path bypasses the simulator. The numbered settings use the training simulator:

| ID | Processing | ID | Processing | ID | Processing |
| --- | --- | --- | --- | --- | --- |
| 0 | Identity | 1 | Gaussian noise | 2 | High-pass filter |
| 3 | Suppression | 4 | EnCodec | 5 | Band-pass filter |
| 6 | Dither | 7 | Amplitude scaling | 8 | Echo |

`test.py` evaluates one setting per run. The paper's 96.22% result aggregates the full neural codec benchmark. For payloads other than 100 bits, supply the corresponding `watermarks/watermark<LENGTH>.pt` and matching checkpoint. Evaluation selects a fixed pool row per vocoder in [runtime.py](runtime.py).

## Acknowledgments

Warden builds on the implementations of [DiffWave](https://github.com/lmnt-com/diffwave), [PriorGrad](https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder), [HiFi-GAN](https://github.com/jik876/hifi-gan), and [BigVGAN](https://github.com/NVIDIA/BigVGAN). The distortion simulator uses [EnCodec](https://github.com/facebookresearch/encodec) and audio effects adapted from [AudioSeal](https://github.com/facebookresearch/audioseal).

## Citation

The arXiv v1 preprint appeared under the name *VocBulwark*.

```bibtex
@misc{liu2026vocbulwark,
  title         = {{VocBulwark}: Towards Practical Generative Speech Watermarking via Additional-Parameter Injection},
  author        = {Weizhi Liu and Yue Li and Zhaoxia Yin},
  year          = {2026},
  eprint        = {2601.22556},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2601.22556v1}
}
```
