# SyntheticSpeechCommands/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from torchaudio import transforms
import torch.nn.functional as F
from torchvision.models import resnet18
from pathlib import Path
import streamlit as st
import soundfile as sf
import torch.nn as nn
import uvicorn
import torch
import io


BASE_DIR = Path(__file__).parent


# ── Model ──────────────────────────────────────────────────────────────────────
class CheckAudio(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ── Transform ──────────────────────────────────────────────────────────────────
mel_transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=160,
    n_mels=80
)
amp_to_db = transforms.AmplitudeToDB()
TARGET_TIME_FRAMES = 101


def change_audio(waveform, sample_rate):
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)

    if sample_rate != 16000:
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

    spec = amp_to_db(mel_transform(waveform)).squeeze(0)

    if spec.shape[1] > TARGET_TIME_FRAMES:
        spec = spec[:, :TARGET_TIME_FRAMES]
    elif spec.shape[1] < TARGET_TIME_FRAMES:
        pad_amount = TARGET_TIME_FRAMES - spec.shape[1]
        spec = F.pad(spec, (0, pad_amount))

    return spec


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = torch.load(BASE_DIR / 'label_SyntheticSpeechCommands.pth', weights_only=False)
    model = CheckAudio(num_classes=len(classes))
    model.load_state_dict(torch.load(BASE_DIR / 'model_CheckAudio_SyntheticSpeechCommands.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device, classes


model, device, classes = load_model()
index_to_label = {ind: lab for ind, lab in enumerate(classes)}


app = FastAPI()


@app.post('/predict')
async def predict_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Файл пустой')

        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        wf = torch.tensor(wf).T if wf.ndim == 2 else torch.tensor(wf)  # (C, T) или (T,)

        spec = change_audio(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = torch.argmax(y_pred, dim=1).item()
            pred_class = index_to_label[pred_ind]
            return {'Индекс': pred_ind, 'Класс': pred_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


# ── Streamlit ──────────────────────────────────────────────────────────────────
# st.title('Synthetic Speech Commands Model')
# st.text('Загрузите аудиофайл с командой, и модель попробует её распознать.')
#
# uploaded_file = st.file_uploader('Выберите аудиофайл', type=['wav', 'flac', 'ogg'])
#
# if not uploaded_file:
#     st.info('Загрузите аудиофайл')
# else:
#     st.audio(uploaded_file)
#
#     if st.button('Распознать команду'):
#         try:
#             wf, sr = sf.read(io.BytesIO(uploaded_file.read()), dtype='float32')
#             wf = torch.tensor(wf).T if wf.ndim == 2 else torch.tensor(wf)  # (C, T) или (T,)
#
#             spec = change_audio(wf, sr).unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 y_pred = model(spec)
#                 prediction = y_pred.argmax(dim=1).item()
#
#             st.success(f'Модель думает, что это команда: {index_to_label[prediction]}')
#
#         except Exception as e:
#             st.error(f'Ошибка: {str(e)}')
