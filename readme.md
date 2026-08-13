# Voice Command Recognition System

> ResNet18 (адаптированный под аудио) классифицирует 35 голосовых команд
> полностью на устройстве — без cloud API, без сетевой задержки,
> без передачи аудио на сторонние серверы.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![torchaudio](https://img.shields.io/badge/torchaudio-2.x-purple)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.1x-teal)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Проблема

Cloud speech API (Google, AWS) стоят $0.004–$0.006 за запрос.
При 1 млн команд/день — это $4 000–$6 000 в месяц плюс 200–800 мс сетевой задержки.
В healthcare, industrial IoT и embedded-устройствах аудио вообще нельзя отправлять наружу.
Этот классификатор работает полностью локально: $0 API-затрат, инференс на CPU,
аудио не покидает устройство.

---

## Быстрый старт

```bash
git clone https://github.com/your-username/voice-command-recognition
cd SyntheticSpeechCommands
pip install -r requirements.txt

# Обучение (Google Colab, см. SyntheticSpeechCommands.ipynb)
# → model_CheckAudio_SyntheticSpeechCommands.pth + label_SyntheticSpeechCommands.pth

# Вариант 1 — REST API
python main.py
# → POST http://127.0.0.1:8000/predict  (multipart/form-data, поле file)

# Вариант 2 — веб-интерфейс с загрузкой файла
streamlit run main.py
```

> В `main.py` оба интерфейса лежат рядом: FastAPI-эндпоинт активен по умолчанию,
> Streamlit-блок закомментирован внизу файла — раскомментируйте его вместо
> `if __name__ == '__main__':`, чтобы поднять веб-демо.

---

## Demo

**FastAPI:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@test audio/five.wav"
# → {"Индекс": 7, "Класс": "five"}
```

**Streamlit:** загрузка WAV/FLAC/OGG → кнопка "Распознать команду" →
"Модель думает, что это команда: five"

**35 поддерживаемых команд:**

stop · go · yes · no · up · down · left · right · forward · backward

on · off · follow · learn · visual · zero–nine (0–9) · happy · wow

bird · cat · dog · bed · house · tree · marvin · sheila

---

## Результаты

| Модель                                  | Accuracy | Параметры | Размер   |
|------------------------------------------|----------|-----------|----------|
| Random classifier (35 классов)            | 2.9%     | —         | —        |
| **ResNet18 adapted (этот проект)**        | **97.15%** (val) | **~11.2M** | **~44.8 МБ** |
| Wav2Vec 2.0 (pretrained, референс)        | ~97%     | 95M       | 360 МБ   |

train: 95.54% · test: 97.03% · val: 97.15% (после 20 эпох, AdamW + CosineAnnealingLR)

**Почему адаптированный ResNet18, а не Wav2Vec 2.0:**
Wav2Vec 2.0 даёт сопоставимую точность, но весит 360 МБ — почти в 8 раз больше.
ResNet18 с 1-канальным первым слоем под Mel-спектрограмму даёт тот же уровень
точности при значительно меньшем футпринте, что критично для CPU-инференса
и edge-деплоя.

---

## Датасет

- **Источник:** Google Speech Commands v2 — `torchaudio.datasets.SPEECHCOMMANDS`
- **Объём:** ~105 000 односекундных WAV-клипов, 16 кГц, 35 классов
- **Feature extraction:** raw waveform → `MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=80)` → `AmplitudeToDB` → тензор `[1, 80, 101]`, обрабатывается как 2D-изображение

| Проблема данных                              | Решение                                                  | Эффект                    |
|------------------------------------------------|-----------------------------------------------------------|----------------------------|
| Клипы переменной длины по времени               | Truncation / `F.pad` до `TARGET_TIME_FRAMES=101`          | Консистентный CNN input   |
| Аугментация на train (SpecAugment)              | `FrequencyMasking(15)` + `TimeMasking(35)`                 | Устойчивость к шуму        |
| Пользовательский аудио произвольной sample rate | `Resample(orig_freq=sr, new_freq=16000)` при инференсе     | Любой формат без ошибок    |
| Стерео на входе                                 | `waveform.mean(dim=0, keepdim=True)` → моно                | Единый формат для модели   |

---

## Архитектура
```
Аудио (WAV/MP3/FLAC/OGG / микрофон)
│
▼
Resample → 16 кГц (если нужно) + приведение к моно
│
▼
MelSpectrogram(n_fft=1024, hop=160, n_mels=80) → AmplitudeToDB
│
▼
truncate / F.pad до TARGET_TIME_FRAMES=101 → [1, 80, 101]
│
▼
ResNet18 (веса не предобучены)
conv1: Conv2d(1→64, kernel=7, stride=2, padding=3) ← адаптирован под 1-канальный вход
... стандартные residual-блоки ResNet18 ...
fc: Linear(512 → 35)
│
▼
argmax → predicted command label
```

**Почему адаптация `conv1`, а не 3-канальное дублирование спектрограммы:**
Вместо копирования Mel-спектрограммы в 3 канала под RGB-вход ResNet,
первый свёрточный слой пересоздан под 1 канал напрямую. Это убирает
избыточные вычисления и держит модель компактной без потери точности.

---

## Стек

| Слой           | Технологии                                       |
|----------------|---------------------------------------------------|
| ML             | PyTorch, torchvision (ResNet18), torchaudio       |
| Audio          | soundfile, `torchaudio.transforms`                |
| API            | FastAPI, uvicorn                                  |
| UI / Demo      | Streamlit                                         |
| Regularization | AdamW, CosineAnnealingLR, SpecAugment             |
| Deploy         | Local CPU / Streamlit Cloud                       |

---

## Что дальше (Roadmap)

- [ ] Разделить FastAPI и Streamlit на отдельные entry-point'ы вместо одного `main.py`
- [ ] `torch.jit.script` / ONNX экспорт → деплой без Python runtime на embedded
- [ ] Fine-tuning на шумных данных → устойчивость в production-среде
- [ ] Wake-word режим: непрерывный стриминг с детекцией hotword
- [ ] MLflow: трекинг экспериментов по аугментациям и архитектурам
- [ ] Docker-образ с обоими интерфейсами

---

## Business Impact

| Сценарий                      | Cloud API             | Этот проект           |
|--------------------------------|------------------------|-------------------------|
| Стоимость при 1M команд/день  | $4 000–$6 000/мес      | $0                       |
| Латентность                    | 200–800 мс (сеть)      | Локальный CPU-инференс  |
| Приватность аудио              | Передаётся на серверы  | Не покидает устройство  |
| Работа офлайн                  | Нет                    | Да                       |


## Структура репозитория
```
ml_SyntheticSpeechCommands/
├── .gitignore
├── readme.md
├── requirements.txt
└── SyntheticSpeechCommands/
    ├── SyntheticSpeechCommands.ipynb
    ├── labels.py
    ├── main.py                                    # FastAPI + Streamlit (закомментирован)
    ├── label_SyntheticSpeechCommands.pth
    ├── model_CheckAudio_SyntheticSpeechCommands.pth
    ├── api/
    │   └── __init__.py
    ├── db/
    │   └── __init__.py
    └── test audio/
        ├── dog.wav
        ├── five.wav
        └── two.wav
```
---