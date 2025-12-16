from ensemble import SpeechEmotionEnsembleModel
import gradio as gr

# from transformers import Wav2Vec2Model
# Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
predictor = SpeechEmotionEnsembleModel(
        "models/cnn_bilstm_state.pt",
        "models/logreg_w2v.joblib",
)

def run_inference(audio_path):
    print('called')
    if audio_path is None:
        return "No audio provided", None

    # Your predict + plot pipeline
    result = predictor.predict(audio_path)
    label, fig = predictor.plot_probs(result)
    print(result)
    return label, fig

with gr.Blocks() as demo:
    gr.Markdown("# Speech Emotion Recognition Demo")

    audio_input = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Input audio (WAV)",
    )

    predict_btn = gr.Button("Predict emotion")

    label_out = gr.Textbox(label="Predicted label")
    plot_out = gr.Plot(label="Probability distribution")

    predict_btn.click(
        fn=run_inference,
        inputs=audio_input,
        outputs=[label_out, plot_out],
    )



if __name__ == "__main__":
    demo.launch()

