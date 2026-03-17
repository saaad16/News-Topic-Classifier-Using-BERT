
import gradio as gr
from transformers import pipeline

model_path = "MuhammadSaad1234/bert-news-classifier"
pipe = pipeline("text-classification", model=model_path)
labels = ["World", "Sports", "Business", "Sci/Tech"]

def classify_news(text):
    if not text.strip(): return {label: 0.0 for label in labels}
    out = pipe(text)[0]
    idx = int(out['label'].split('_')[-1])
    return {labels[idx]: float(out['score'])}

demo = gr.Interface(
    fn=classify_news, 
    inputs=gr.Textbox(label="Headline"), 
    outputs=gr.Label(num_top_classes=4),
    title="News Topic Classifier"
)
if __name__ == "__main__":
    demo.launch()
