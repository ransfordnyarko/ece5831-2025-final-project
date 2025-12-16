import torch
import torch.nn.functional as F
import numpy as np
import librosa
import joblib
from transformers import Wav2Vec2Model
from CNNBiLSTM import CNNBiLSTM_Model
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import gradio as gr
import matplotlib.pyplot as plt


class SpeechEmotionEnsembleModel:
    def __init__(self, cnn_bilstm_path, wav2vec2_path, num_classes=6):
        self.num_classes = num_classes
        logreg, processor, w2v_model = self.load_wav2vec2_model(wav2vec2_path)
        self.cnn_bilstm = self.load_cnn_model(cnn_bilstm_path)
        self.logreg = logreg
        self.processor = processor
        self.w2v_model = w2v_model
        
        self.ID_2_LABEL = {
            0: "ANG",
            1: "DIS",
            2: "FEA",
            3: "HAP",
            4: "NEU",
            5: "SAD",
        }


    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out = (out1 + out2) / 2
        return out

    def load_audio(self, path, sr=16000):
        y, s = librosa.load(path, sr=sr, mono=True)
        return y, s


    def trim_and_normalize(self, y, top_db=30, target_dbfs=-1.0):
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        peak = np.max(np.abs(yt)) + 1e-9
        scale = (10 ** (target_dbfs / 20.0)) / peak
        return yt * min(scale, 1.0)


    def to_logmel(self, y, sr=16000, n_mels=64):
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            fmin=50,
            fmax=8000,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db.astype(np.float32)

    
    def load_cnn_model(self, path):
        model = CNNBiLSTM_Model(self.num_classes, n_mels=64)
        state = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state["state_dict"])
        return model

    def predict_cnn(self, audio_path):

        y, sr = self.load_audio(audio_path)
        y = self.trim_and_normalize(y)
        logmel = self.to_logmel(y, sr=sr)

        x = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = self.cnn_bilstm(x)
            probs = F.softmax(logits, dim=-1)

        return probs
    
    def load_wav2vec2_model(self, path):
        logreg = joblib.load(path)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        return logreg, processor, w2v_model
    
    def predict_wav2vec(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000)

        inputs = self.processor(y, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            outputs = self.w2v_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).numpy()

        probs = self.logreg.predict_proba(emb)[0]
        return probs
    
    def predict(self, audio_path, alpha=0.25):
        
        probs_cnn = self.predict_cnn(audio_path).squeeze().numpy()
        probs_wav2vec = self.predict_wav2vec(audio_path)

        if probs_cnn.shape != probs_wav2vec.shape:
            raise ValueError("Probability shapes do not match for ensemble.")
        
        final_probs = (1 - alpha) * probs_cnn + alpha * probs_wav2vec
        pred_id = final_probs.argmax()


        return {
            "label": self.ID_2_LABEL[pred_id],
            "probabilities": {
                self.ID_2_LABEL[i]: float(final_probs[i]) for i in range(self.num_classes)
            }
        }

    def plot_probs(self, result):
        label = result["label"]
        probs = result["probabilities"]

        emotions = list(probs.keys())
        values = [probs[e] for e in emotions]

        fig, ax = plt.subplots()
        ax.bar(emotions, values)
        ax.set_title(f"Emotion probabilities (predicted: {label})")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.0)

        return label, fig
