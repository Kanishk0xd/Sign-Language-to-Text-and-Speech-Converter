# Sign Language to Text and Speech

Real-time sign language recognition that converts signs into readable text and natural-sounding speech. Built in Python with a computer vision pipeline for keypoint tracking and a sequence model for gesture classification. Designed to support accessible communication between Deaf/Hard-of-Hearing signers and non-signers.

> Note: Accuracy varies by signer, lighting, camera angle, and dialect. This tool complements but does not replace professional interpreters.

## Features

- Real-time sign recognition from a webcam or video file
- Live text captions + optional synthesized speech (TTS)
- On-device inference for privacy (no video leaves your machine)
- Configurable vocabulary and custom phrase mapping
- Optional offline TTS (pyttsx3) or online TTS (gTTS)
- Extensible model interface (LSTM/Transformer or your own)

## Demo

- Live demo (webcam): `python app.py --source webcam`
- From a video file: `python app.py --source video.mp4`

[Add a GIF or short video demo here once available]

## Architecture

1. Frame capture (webcam or file)
2. Keypoint extraction (e.g., MediaPipe Hands/Holistic or OpenPose)
3. Temporal sequence modeling (e.g., LSTM/GRU/Transformer)
4. Post-processing for sentence smoothing and phrase mapping
5. Text output + optional Text-to-Speech (pyttsx3 or gTTS)

```
Camera → Keypoints → Sequence Model → Language/Post-processing → Text & Speech
```

## Supported Scope

- Default: ASL basics (configure or expand with your dataset)
- Extendable to other sign languages with appropriate datasets and retraining
- Multilingual TTS output (depends on TTS engine)

## Getting Started

### Prerequisites

- Python 3.9+ (recommended)
- pip or conda
- A working webcam (for real-time mode)

### Installation

```bash
git clone https://github.com/<your-org-or-user>/<your-repo>.git
cd <your-repo>

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Typical dependencies (adjust to your stack):
- opencv-python
- mediapipe
- numpy
- torch or tensorflow (choose one)
- pyttsx3 or gTTS (choose one)
- pyyaml (for config)

### Model Weights

Place your trained model weights in:
```
models/
  - asl_sequence_model.pt        # for PyTorch
  # or
  - asl_sequence_model.h5        # for TensorFlow/Keras
```

Update `config.yaml` with the correct path.

## Usage

### Quick Start

- Use webcam (default device 0):
```bash
python app.py --source webcam
```

- Use a video file:
```bash
python app.py --source path/to/video.mp4
```

- Enable speech output:
```bash
python app.py --source webcam --tts pyttsx3
# or
python app.py --source webcam --tts gtts --tts-lang en
```

- Set confidence thresholds and smoothing:
```bash
python app.py --source webcam --min-conf 0.7 --smooth 5
```

### Common Arguments

- `--source`            webcam | <path/to/video>
- `--model`             models/asl_sequence_model.pt (or .h5)
- `--labels`            data/labels.json
- `--tts`               none | pyttsx3 | gtts
- `--tts-lang`          en | hi | es | ...
- `--voice`             voice name/index (if supported by engine)
- `--min-conf`          classification confidence threshold (0–1)
- `--smooth`            window size for temporal smoothing
- `--show`              display annotated frames (on by default)
- `--no-show`           disable window display (headless)

## Configuration

An example `config.yaml`:

```yaml
video:
  source: webcam   # or path/to/video.mp4
  width: 640
  height: 480
  fps: 30

keypoints:
  backend: mediapipe  # or openpose
  holistic: true
  hands_only: false

model:
  framework: torch    # or tensorflow
  weights: models/asl_sequence_model.pt
  labels: data/labels.json
  min_conf: 0.7
  smoothing: 5

tts:
  engine: pyttsx3     # none | pyttsx3 | gtts
  language: en
  voice: default
  rate: 175

postprocess:
  phrase_map: data/phrases.json  # map short tokens → full phrases
```

## Training (Optional)

1. Collect or download a sign dataset (ensure licensing and consent).
2. Extract keypoints per frame/sequence (store as `.npy` or `.csv`).
3. Train a temporal model (LSTM/GRU/Transformer):
   - Input: sequences of keypoints (hands/body/face as applicable)
   - Output: sign class or token sequence
4. Export weights and update `config.yaml`.

Example training outline (PyTorch):
```bash
python tools/extract_keypoints.py --dataset data/raw --out data/keypoints
python tools/train.py --data data/keypoints --labels data/labels.json --out models/asl_sequence_model.pt
```

## Datasets

- [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [WLASL](https://dxli94.github.io/WLASL/)
- [LSA64](http://facundoq.github.io/datasets/lsa64/)
- Your own curated dataset with documented consent

Ensure compliance with dataset licenses and privacy.

## Limitations and Ethics

- Not a substitute for certified interpreters.
- Accuracy depends on lighting, camera quality, signer style, speed, and dialect.
- Obtain consent before recording. Prefer on-device processing.
- Dataset bias may reduce accuracy for some users. Contributions welcome.

## Roadmap

- [ ] Expand vocabulary and phrase coverage
- [ ] Multi-signer robustness and domain adaptation
- [ ] Better sentence-level decoding and punctuation
- [ ] Noise/lighting augmentation for robustness
- [ ] Multilingual UI and captions
- [ ] Export transcripts and session logs (opt-in)

## Contributing

Contributions are welcome! Please open an issue to discuss major changes, and ensure tests and docs accompany PRs.

### Development Setup

```bash
pip install -r requirements-dev.txt
pre-commit install
pytest -q
```

## License

[MIT](LICENSE) 

## Acknowledgements

- Inspired by community work on sign recognition, including MediaPipe/OpenCV pipelines and sequence models similar to [Devansh-47/Sign-Language-To-Text-and-Speech-Conversion](https://github.com/Devansh-47/Sign-Language-To-Text-and-Speech-Conversion).
- Thanks to contributors and dataset authors whose work makes this possible.

## Citation

If you use this project in academic work, please cite it appropriately and include dataset citations as required.

## Contact

For questions or collaboration, please open an issue or reach out: kanishtkhakur115@gmail.com
