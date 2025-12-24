# Download.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
SAVE_DIR = "./Emotion_Analysis_System/emotion_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print("Emotion model downloaded and saved locally.")
