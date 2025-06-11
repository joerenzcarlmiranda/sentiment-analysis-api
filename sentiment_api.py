from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
import re
from textblob import TextBlob
import requests
import os

app = Flask(__name__)

# Your existing Pangasinan dictionaries remain the same
pangasinan_sentiment_dict = {
    # Positive words/phrases
    "mabli": "positive",
    "maong": "positive",
    "masanten": "positive",
    "maliket": "positive",
    "salamat": "positive",
    "makapaliket": "positive",
    "mabulos": "positive",
    "magayaga": "positive",
    "masanti": "positive",  # fixed typo
    "marakep": "positive",  # fixed typo
    "maples": "positive",  # fixed typo
    "masantos": "positive",  # fixed typo

    # Negative words/phrases
    "mauges": "negative",
    "masakit": "negative",
    "onsot": "negative",
    "amta la": "negative",
    "maermen": "negative",
    "mabayag": "negative",
    "anggapo": "negative",
    "mainomay": "negative",
    "makapabwesit": "negative",
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
    """Check for Pangasinan words and expressions to determine sentiment."""
    text_lower = text.lower()

    # Check for expressions first
    for expression, sentiment in pangasinan_expressions.items():
        if expression.lower() in text_lower:
            return sentiment, True

    # Check for individual words
    for word, sentiment in pangasinan_sentiment_dict.items():
        if re.search(r'\b' + re.escape(word.lower()) + r'\b', text_lower):
            return sentiment, True

    return None, False


def detect_and_translate(text):
    """Translate text to English with special handling for Pangasinan."""
    pangasinan_sentiment, is_pangasinan = check_pangasinan_sentiment(text)

    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        return translated_text, pangasinan_sentiment, is_pangasinan
    except Exception as e:
        return text, pangasinan_sentiment, is_pangasinan


def lightweight_sentiment_analysis(text):
    """Use TextBlob for lightweight sentiment analysis."""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    except:
        return "neutral"


def enhanced_sentiment_analysis(text):
    """Enhanced sentiment analysis with multiple fallback methods."""
    # Method 1: Try Hugging Face API (if token is available)
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        try:
            API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
            headers = {"Authorization": f"Bearer {hf_token}"}

            response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    top_result = max(result[0], key=lambda x: x['score'])
                    label = top_result['label'].lower()

                    if 'positive' in label or label == 'label_2':
                        return "positive"
                    elif 'negative' in label or label == 'label_0':
                        return "negative"
                    else:
                        return "neutral"
        except:
            pass

    # Method 2: TextBlob sentiment analysis
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    except:
        pass

    # Method 3: Simple keyword-based fallback
    text_lower = text.lower()
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy', 'awesome']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'angry', 'disappointed', 'horrible', 'worst']

    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        text = data.get("text", "")

        # Check for Pangasinan sentiment first
        translated_text, pangasinan_sentiment, is_pangasinan = detect_and_translate(text)

        if pangasinan_sentiment:
            sentiment = pangasinan_sentiment
        else:
            # Use enhanced sentiment analysis with multiple fallbacks
            sentiment = enhanced_sentiment_analysis(translated_text)

        response = {
            "Feedback": text,
            "Sentiment Result": sentiment,
            "translated_text": translated_text,
            "contains_pangasinan": is_pangasinan
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Pangasinan Sentiment API is running"})


if __name__ == '__main__':
    app.run(debug=True)