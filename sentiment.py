from textblob import TextBlob
from deep_translator import GoogleTranslator  # Replaces googletrans


def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 to 1 (negative to positive)

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


def translate_text(text, target_lang="en"):
    """Translate text using deep_translator instead of googletrans"""
    translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
    return translated
