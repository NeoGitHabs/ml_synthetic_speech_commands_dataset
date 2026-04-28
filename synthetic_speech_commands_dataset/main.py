from fastapi import FastAPI, UploadFile, File, HTTPException
from torchaudio import transforms
import torch.nn.functional as F
import soundfile as sf
import streamlit as st
import torch.nn as nn
import tempfile
import uvicorn
import torch
import os
import io



classes = ['bed', 'four', 'happy', 'nine', 'zero', 'go', 'eight', 'down', 'house', 'bird',
           'off', 'marvin', 'on', 'up', 'sheila', 'visual', 'follow', 'cat', 'yes', 'tree',
           'learn', 'six', 'no', 'left', 'one', 'two', 'stop', 'seven', 'backward', 'three',
           'dog', 'wow', 'five', 'forward', 'right']


class CheckAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 35)  # 35 классов
        )

    def forward(self, audio):
        audio = audio.unsqueeze(1)
        audio = self.first(audio)
        audio = self.second(audio)
        return audio


index_to_label = {ind:lab for ind, lab in enumerate(classes)}

transform = transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
max_len = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CheckAudio()
model.load_state_dict(torch.load('audioCpeechCommands_model.pth', map_location=device))
model.to(device)
model.eval()

# app = FastAPI()
#
# def change_audio(waveform, sample_rate):
#     if sample_rate != 16000:
#         new_sr = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#         waveform = new_sr(torch.tensor(waveform))
#
#     spec = transform(waveform).squeeze(0)
#
#     if spec.shape[1] > max_len:
#         spec = spec[:, :max_len]
#
#     if spec.shape[1] < max_len:
#         count_len = max_len - spec.shape[1]
#         spec = F.pad(spec, (0, count_len))
#
#     return spec
#
#
# @app.post('/predict')
# async def predict_audio(file: UploadFile = File(...)):
#     try:
#         data = await file.read()
#         if not data:
#             raise HTTPException(status_code=400, detail='Файл пустой')
#
#         wf, sr = sf.read(io.BytesIO(data), dtype='float32')
#         wf = torch.tensor(wf).T
#
#         spec = change_audio(wf, sr).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             y_pred = model(spec)
#             pred_ind = torch.argmax(y_pred, dim=1).item()
#             pred_class = index_to_label[pred_ind]
#             return {'Индекс': pred_ind, 'Класс': pred_class}
#
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)


st.title('Speech Commands Classifier')
st.text('Загрузите аудио команду или голосовую команду, и модель попробует её распознать.')

audio_file = st.file_uploader('Выберите аудио', type=['wav', 'mp3', 'flac', 'ogg'])

speech_file = st.audio_input('Говорить команду: ')

if not audio_file:
    st.info('Загрузите аудио или говорите команду')
else:
    st.audio(audio_file) or st.audio(speech_file)

    if st.button('Распознать'):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name

            # Загрузка через soundfile
            waveform, sr = sf.read(tmp_path)
            waveform = torch.from_numpy(waveform).float()

            # Преобразование в формат [channels, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T

            os.unlink(tmp_path)

            # Ресемплинг если нужно
            if sr != 16000:
                resampler = transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            # Спектрограмма + паддинг
            spec = transform(waveform).squeeze(0)

            if spec.shape[1] > max_len:
                spec = spec[:, :max_len]
            elif spec.shape[1] < max_len:
                count_len = max_len - spec.shape[1]
                spec = F.pad(spec, (0, count_len))

            spec = spec.unsqueeze(0).to(device)

            with torch.no_grad():
                y_prediction = model(spec)
                prediction = y_prediction.argmax(dim=1).item()

            st.success(f'Модель думает, что это команда: {classes[prediction]}')

        except Exception as e:
            st.error(f'Ошибка: {str(e)}')
