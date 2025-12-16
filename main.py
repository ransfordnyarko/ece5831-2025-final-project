import os
import sys
from ensemble import SpeechEmotionEnsembleModel

def parse_true_emotion(path):
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[-1]
    return None


TRUE_LABEL_2_PRED_LABEL = {
    "anger": "ANG",
    "happy": "HAP",
    "sad": "SAD",
    "neutral": "NEU",
    "fea": "FEA",
    "disgust": "DIS",
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/name_emotion.wav")
        sys.exit(1)
    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        sys.exit(1)
    true_emotion = parse_true_emotion(wav_path)
    if true_emotion is None:
        print("Could not parse true emotion from filename. Expected format: name_emotion.wav")
        sys.exit(1)

    predictor = SpeechEmotionEnsembleModel(
        "models/cnn_bilstm_state.pt",
        "models/logreg_w2v.joblib",
    )

    result = predictor.predict(wav_path)
    label, fig = predictor.plot_probs(result)

    match = (label == TRUE_LABEL_2_PRED_LABEL[true_emotion])
    
    print(f"True emotion: {TRUE_LABEL_2_PRED_LABEL[true_emotion]}")
    print(f"Predicted   : {label}")
    print("Correct     : {}".format("YES" if match else "NO"))

if __name__ == '__main__':
    main()