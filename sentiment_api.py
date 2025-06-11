from transformers import pipeline
from deep_translator import GoogleTranslator
import re
import gradio as gr

# Load Hugging Face sentiment analysis pipeline (uses remote API if model too large)
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Pangasinan dictionary
pangasinan_sentiment_dict = {
    "mabli": "positive",
    "maong": "positive",
    "masanten": "positive",
    "maliket": "positive",
    "salamat": "positive",
    "makapaliket": "positive",
    "mabulos": "positive",
    "magayaga": "positive",
    "masanting": "positive",
    "marakep": "positive",
    "maples": "positive",
    "masantos": "positive",
    "mauges": "negative",
    "masakit": "negative",
    "onsot": "negative",
    "amta la": "negative",
    "maermen": "negative",
    "mabayag": "negative",
    "anggapo": "negative",
    "mainomay": "negative",
    "makapabwesit": "negative",
    "sankaili": "neutral",
    "onla": "neutral",
    "mansiansia": "neutral",
    "mankakasi": "neutral"
}

pangasinan_expressions = {
    "anggapo so nakala": "negative",
    "masakit so ulok": "negative",
    "maong ya agew": "positive",
    "mabayag so pila": "negative",
    "masanten ya bulan": "positive",
    "masakbay ka la": "neutral",
    "maong so ginawam": "positive",
    "maermen ak": "negative",
    "maliket ak": "positive",
    "salamat na dakel": "positive",
    "makapabwesit so office u": "negative",
    "masantos na kabwasan sikayo amin": "positive"
}


def check_pangasinan_sentiment(text):
    text_lower = text.lower()
    for expression, sentiment in pangasinan_expressions.items():
        if expression in text_lower:
            return sentiment, True
    for word, sentiment in pangasinan_sentiment_dict.items():
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return sentiment, True
    return None, False


def detect_and_translate(text):
    pangasinan_sentiment, is_pangasinan = check_pangasinan_sentiment(text)
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        translated_text = text
    return translated_text, pangasinan_sentiment, is_pangasinan


def custom_sentiment_logic(translated_text):
    text_lower = translated_text.lower()
    if any(phrase in text_lower for phrase in ["long queue", "long wait", "waiting for hours"]):
        return "negative"
    if any(phrase in text_lower for phrase in ["excellent service", "wonderful experience"]):
        return "positive"
    return None


def analyze(text):
    translated_text, pangasinan_sentiment, is_pangasinan = detect_and_translate(text)

    if pangasinan_sentiment:
        sentiment = pangasinan_sentiment
    else:
        custom_sentiment = custom_sentiment_logic(translated_text)
        if custom_sentiment:
            sentiment = custom_sentiment
        else:
            result = sentiment_pipeline(translated_text)
            sentiment_label = result[0]['label'].lower()
            sentiment_map = {
                "label_0": "negative",
                "label_1": "neutral",
                "label_2": "positive",
                "negative": "negative",
                "neutral": "neutral",
                "positive": "positive"
            }
            sentiment = sentiment_map.get(sentiment_label, "neutral")

    return {
        "Original Text": text,
        "Translated Text": translated_text,
        "Sentiment": sentiment,
        "Contains Pangasinan Words": is_pangasinan
    }


iface = gr.Interface(fn=analyze,
                     inputs=gr.Textbox(label="Enter Text (Pangasinan/English)"),
                     outputs="json",
                     title="Pangasinan-Aware Sentiment Analysis",
                     description="Supports English and Pangasinan. Translates, detects local expressions, and predicts sentiment.")

if __name__ == "__main__":
    iface.launch(share=True)
