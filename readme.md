# Voice Command Recognition System

> A CNN-powered speech classifier that identifies 35 spoken commands in
> real time from microphone or audio file input — enabling hands-free
> control interfaces, accessibility tools, and voice-activated automation
> without cloud API dependencies.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![torchaudio](https://img.shields.io/badge/torchaudio-2.x-purple)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-~87%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Business Problem

Voice-controlled interfaces are in high demand across smart home devices,
industrial automation, and accessibility software — yet most solutions
depend on expensive cloud speech APIs (Google, AWS) with per-request
costs that scale poorly at volume. An on-device keyword spotter that runs
entirely locally cuts API costs to zero, eliminates latency from network
round-trips, and keeps sensitive audio data off third-party servers —
critical for healthcare, finance, and industrial IoT deployments.

---

## Demo

Launch the app and either upload an audio file or speak directly into
your microphone:

```bash
streamlit run main.py
```

**App flow:**
1. Upload a WAV / MP3 / FLAC / OGG file **or** click the microphone button
2. Click **"Распознать"**
3. Model returns the predicted spoken command

**Example output:**
```
✅ Модель думает, что это команда: stop
```

**Supported commands (35 total):**
`stop · go · yes · no · up · down · left · right · forward · backward ·
on · off · follow · learn · visual · zero–nine · happy · wow · bird ·
cat · dog · bed · house · tree · marvin · sheila`

---

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~87%   |
| F1-score  | ~0.87  |
| Precision | ~0.88  |
| Recall    | ~0.87  |

Best model: AudioSpeechCNN — 3-block CNN on Mel spectrograms
Baseline (random classifier, 35 classes): Accuracy = 2.9%
↑ +84.1% improvement vs baseline

> Note: Wav2Vec 2.0 (pretrained) achieves ~97% on this benchmark.
> This model reaches ~87% trained entirely from scratch in 15 epochs —
> no pretrained weights, no cloud APIs.

---

## Dataset

- **Source:** Google Speech Commands v2 (torchaudio.datasets.SPEECHCOMMANDS)
- **Size:** ~105,000 one-second WAV clips, 16kHz mono
- **Features:** Raw waveform → Mel spectrogram (64 mel bins × 81 time frames)
  treated as a single-channel 2D image for CNN input
- **Class balance:** Near-balanced — ~3,000 clips per command class;
  audio samples shorter than exactly 1 second filtered out in `collate_fn`
  to ensure consistent tensor shape across batches

---

## Approach

1. **Data Loading** — Streamed via `torchaudio.datasets.SPEECHCOMMANDS`
   with official `training` / `testing` splits; `batch_size=256`
2. **Feature Extraction** — Raw waveform → `MelSpectrogram`
   (sample_rate=16000, n_mels=64) → output shape `[1, 64, 81]`;
   treated as a single-channel image for 2D CNN processing
3. **Batch Filtering** — Custom `collate_fn` filters clips ≠ 16,000 samples
   to ensure fixed-length input; empty batches skipped during training
4. **Inference Preprocessing** — Sample rate normalization via
   `torchaudio.transforms.Resample` for non-16kHz uploads;
   spectrogram padding/truncation to `max_len=100` frames via `F.pad`
5. **Model Architecture** — 3-block CNN:
   `Conv2d(1→32→64→128)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2)` →
   `AdaptiveAvgPool2d((8,8))` for inference robustness →
   `Linear(2048→128)` + `Linear(128→35)`
6. **Training** — 15 epochs, AdamW (lr=0.001, weight_decay=1e-4),
   CrossEntropyLoss, GPU-accelerated
7. **Deployment** — Streamlit UI with dual input: file upload
   (WAV/MP3/FLAC/OGG) + live microphone via `st.audio_input`;
   temp file handling via `tempfile` + `os.unlink`

---

## Key Challenges & Solutions

**Variable-length audio causing tensor shape mismatches in batching**
Raw audio clips in the dataset vary from 0.5 to 1.0 seconds —
standard `DataLoader` collation fails with mixed-length tensors →
implemented custom `collate_fn` that filters to exactly 16,000-sample
clips and skips empty batches during training → zero shape-mismatch
errors across 15 epochs of training on 105,000 clips.

**Sample rate mismatch for real-world audio uploads**
User-uploaded files can be 8kHz, 22kHz, 44kHz, or 48kHz — the model
expects 16kHz Mel spectrograms → added automatic sample rate detection
with `torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)`
in both the inference API and Streamlit pipeline → model handles any
standard audio format without preprocessing errors.

**Spectrogram length variance during live microphone inference**
Microphone recordings vary in length depending on how long the user
speaks — feeding variable-length spectrograms to a fixed-input CNN
causes dimension errors → added spectrogram truncation
(`spec[:, :max_len]`) and zero-padding (`F.pad(spec, (0, count_len))`)
to normalize all inputs to `max_len=100` frames →
consistent inference behavior across both uploaded files and
live microphone input.

---

## Tech Stack

| Category      | Tools                                        |
|---------------|----------------------------------------------|
| Language      | Python 3.11                                  |
| ML            | PyTorch, torchaudio                          |
| Audio         | soundfile, torchaudio.transforms             |
| UI / Demo     | Streamlit (`st.audio_input` for mic)         |
| Regularization| BatchNorm2d, Dropout, AdamW, weight_decay    |
| Deploy        | Streamlit (local / Streamlit Cloud)          |

---

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/your-username/voice-command-recognition
cd voice-command-recognition
pip install torch torchaudio streamlit soundfile matplotlib
```

```bash
# 2. Train the model (saves audioSpeechCommands_model.pth + label.pth)
python train.py
```

```bash
# 3. Launch the web app
streamlit run main.py
```

---

## Business Impact

- ↓ ~100% reduction in cloud speech API costs vs Google/AWS keyword
  spotting for on-device deployments (estimated)
- ↑ ~87% on-device recognition accuracy for 35 command vocabulary —
  sufficient for hands-free control in low-noise environments (estimated)
- ↓ ~80% reduction in voice command response latency vs cloud-dependent
  solutions due to local inference with no network round-trip (estimated)
- ↑ Dual input (file upload + live microphone) enables both batch
  audio processing pipelines and real-time interactive deployments
- ↑ Fully portable: model runs on CPU with no GPU requirement —
  deployable on Raspberry Pi, Jetson Nano, and edge IoT devices

---

[//]: # (## Author)

[//]: # (Your Name — [LinkedIn]&#40;#&#41; | [GitHub]&#40;#&#41;)