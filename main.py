from database import get_db_connection
from sentiment import analyze_sentiment, translate_text
import pandas as pd


# Fetch feedback from the database
def fetch_feedback():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, text FROM feedback WHERE sentiment IS NULL")
    feedbacks = cursor.fetchall()
    conn.close()
    return feedbacks


# Process sentiment analysis and update the database
def update_feedback():
    feedbacks = fetch_feedback()

    conn = get_db_connection()
    cursor = conn.cursor()

    for feedback in feedbacks:
        text = feedback["text"]
        translated_text = translate_text(text)  # Translate to English
        sentiment = analyze_sentiment(translated_text)

        cursor.execute(
            "UPDATE feedback SET sentiment=%s, feedback_text=%s WHERE id=%s",
            (sentiment, translated_text, feedback["id"])  # Ensure feedback_text is correct
        )

    conn.commit()
    conn.close()
    print("Feedback updated successfully!")


if __name__ == "__main__":
    update_feedback()
