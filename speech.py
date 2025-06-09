# translate.py
import sys
from transformers import MarianMTModel, MarianTokenizer

# Get input text from command-line arguments
if len(sys.argv) < 2:
    print("No input text provided.")
    sys.exit(1)

input_text = sys.argv[1]

# Define source and target languages
src_lang = "tl"  # Tagalog
tgt_lang = "en"  # English
model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

# Load model and tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Tokenize and translate
tokens = tokenizer.prepare_seq2seq_batch([input_text], return_tensors="pt")
translation = model.generate(**tokens)
translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]

# Output translation
print(translated_text)
