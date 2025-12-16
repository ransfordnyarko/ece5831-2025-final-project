# Speech Emotion Recognition using CNN–BiLSTM and wav2vec Embeddings

This project focuses on **speech emotion recognition (SER)** using two complementary modeling approaches:
1. A **task-specific CNN–BiLSTM** model trained on log-mel spectrograms.
2. A **self-supervised learning (SSL)–based approach** using pretrained wav2vec embeddings with a lightweight classifier.

The goal is to demonstrate how combining time–frequency modeling with pretrained speech representations can improve robustness in emotion prediction.

---

## Project Overview

Speech emotion recognition aims to identify the emotional state of a speaker from audio signals.

In this project:
- The **CNN–BiLSTM** model captures local acoustic patterns and long-range temporal dynamics from spectrograms.
- The **wav2vec-based model** leverages representations learned from large-scale unlabeled speech data that transfer well to emotion recognition.
- The final system demonstrates how both approaches independently predict emotion and can be compared or combined.

---

## Model Architectures

### CNN–BiLSTM
- Input: Log-mel spectrograms
- CNN layers extract local time–frequency features
- BiLSTM models temporal evolution of emotional cues
- Final MLP outputs emotion probabilities

### wav2vec Embeddings
- Pretrained self-supervised speech model
- Audio mapped to high-dimensional embeddings
- Lightweight classifier trained for emotion labels

---

## Dataset

- **Dataset used:** CREMA_D
- **Emotion classes:** [Happy, Sad, Angry, Neutral, Disgust, Fead]
- **Preprocessing:**
  - Audio normalization
  - Log-mel spectrogram extraction (for CNN–BiLSTM)
  - wav2vec embedding extraction (for SSL model)
- **Splits:** Train / Validation / Test

Dataset access:  
**https://www.kaggle.com/datasets/ejlok1/cremad**

---

## Results and Evaluation

- Metrics used: Accuracy, Weighted F1-score
- Both models produce probability distributions over emotion classes
- Performance comparison highlights strengths of task-specific vs pretrained representations

(See final report for detailed quantitative results.)

---

## Demo

A live demo illustrates:
- Audio upload or recording
- Emotion prediction
- Probability distribution over emotion classes

**Demo video:**  
[Link to demo video]:https://youtu.be/ZGbZhUc7IHA

---

## Project Deliverables

All required submission materials are linked below:

**Pre-recorded presentation video:**  
[Link to presentation video]: 

**Presentation slides:**  
[Link to presentation slide]: (https://docs.google.com/presentation/d/1XcLfverBRftLosu7dHRQmeapMuWqvffv/edit?usp=drive_link&ouid=104540571341922337102&rtpof=true&sd=true)

**Final report:**  
[Link to report PDF]: https://drive.google.com/file/d/1sAwD3tZmWi7KtlGNAGkWbXO8oPoXPjpx/view?usp=drive_link

- 📊 **Dataset:**  
[Link to dataset or dataset README]: https://www.kaggle.com/datasets/ejlok1/cremad

**Demo video:**  
[Link to demo video]: https://youtu.be/ZGbZhUc7IHA

---

## How to Run (Optional)

If running locally:
1. Install dependencies listed in `requirements.txt`
2. Download models from this link: https://drive.google.com/drive/folders/1pZ9L0hwfDTSE0kUYgCGsQpEKUdla5Zcg?usp=drive_link 
3. Add to root folder
4. Two ways of testing:
    a. Run python demo.py to run gradio server
    b. Run python main.py file_name to test file using script

---


---

## Authors

- [Ransford Nyarko / Dharshani Anandkumar]
- Course: [ECE5831 - Pattern Recognition]


