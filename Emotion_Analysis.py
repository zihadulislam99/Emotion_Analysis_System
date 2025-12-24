from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./Emotion_Analysis_System/emotion_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()  # IMPORTANT for inference

emotion_map = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

def predict_emotion(texts, threshold=0.5):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)

    results = []
    for prob_vector in probs:
        emotions = [
            emotion_map[i]
            for i, score in enumerate(prob_vector)
            if score >= threshold
        ]
        results.append(emotions if emotions else ["neutral"])

    return results


texts = ["I will kill you", "I love learning AI", "I feel scared and nervous", "Thank you so much for your help", "Everything is pointless"]

print(predict_emotion("it's being too difficult")[0])
# for text, emotions in zip(texts, predict_emotion(texts)):
#     print(f"Text: {text}\nmotions: {emotions[0]}\n")
