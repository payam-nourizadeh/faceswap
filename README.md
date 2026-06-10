# Face Swap

Takes a face from one photo and puts it onto the person in another photo or
video, keeping the original pose, expression, and lighting so it looks natural.

Built as a take-home for a computer vision role. It's a SimSwap-style
GAN: one network takes a target face plus a source identity and produces the
swap in a single pass, so it works on any pair of faces without retraining.

## Two things worth knowing up front

The dataset used here (LFW) could be good for *testing* a face swapper but not great
for *training* one; most people in it have only one photo, at low resolution.
So I use it to measure identity preservation (its original purpose) and am
upfront that training data is the real limit on quality. The dataset-update
step exists because better data is the biggest lever.

This is deepfake technology built on real people. It is released for research
and educational use only (non-commercial). Do not use it to create deceptive
media of real people without their consent.

## Setup

```bash
conda env create -f environment.yml
conda activate faceswap
pip install -e .

# then install torch for your machine, check your cuda version if using NVIDIA GPU:
#   NVIDIA:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#   Mac:     pip install torch torchvision
```

You also need pretrained ArcFace weights (r50 for training, r100 for
evaluation). Without them the identity loss is meaningless. I have included
them in the `eval_weights/` folder. The trained model is in `runs/`.

## Usage

```bash
# 1. prepare the data
python scripts/download_data.py --dest data
python scripts/build_manifest.py --config configs/default.yaml

# 2. train
python -m faceswap.train --config configs/default.yaml

# 3. evaluate (writes report + graphs)
python -m faceswap.evaluate --config configs/default.yaml --checkpoint 

# 4. swap an image (checkpoints in runs/). Sample images in data/image_swap/.
#    CPU is the default device; change to cuda src/faceswap/data/alignment.py line 181. To be added to config later
python -m faceswap.api cli --checkpoint  --source s.jpg --target t.jpg --out out.jpg

# 5. swap a video. Sample output in data/video/.
#    Note: video quality is limited by the low-res LFW training data.
python -m faceswap.api video --checkpoint  --source s.jpg --input in.mp4 --out out.mp4

# 6. HTTP service (not yet implemented)
# python -m faceswap.api serve --checkpoint 

# add more data later (deduplicated automatically)
python scripts/build_manifest.py --config configs/default.yaml --update path/to/images
```

## Layout

```
configs/        all settings in one yaml
scripts/        data download + manifest building
src/faceswap/
  data/         align, quality-filter, manifest, pairing, dataset
  models/       generator, discriminator, ArcFace identity encoder
  losses/       identity, reconstruction, adversarial, feature-matching
  evaluation/   metrics + graphs
  train.py      training loop
  evaluate.py   metric suite
  inference.py  full detect -> align -> swap -> blend pipeline
  video.py      per-frame video swapping
  api.py        CLI + HTTP service
```

## Notes

Trains on an NVIDIA GPU, develops/runs on an Apple-silicon Mac, or CPU - set the
device in the config. Output is 256px, chosen to match LFW's real detail rather
than the hardware limit. With more time the priorities are better training data,
higher resolution, and a face-parsing mask for cleaner blends.

