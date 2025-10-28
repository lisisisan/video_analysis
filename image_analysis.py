import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
import cv2
from collections import Counter
import os
import spacy

nlp = spacy.load("en_core_web_sm")

# Загрузка предобученной модели ResNet50
model = resnet50(weights='DEFAULT')
model.eval()

# Трансформация кадра
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Метки ImageNet
from torchvision.models import ResNet50_Weights
labels = ResNet50_Weights.DEFAULT.meta['categories']

# Папка для сохранения кадров с подписями
os.makedirs("image_frames", exist_ok=True)

def predict_frame(frame):
    img = transform(frame).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top5 = torch.topk(probs, 5)
        return [labels[i] for i in top5.indices[0]]

# Открываем видео
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

all_predictions = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preds = predict_frame(frame)
    all_predictions.extend(preds)

    # Сохраняем каждый 30-й кадр с подписью для визуализации
    if frame_count % 30 == 0:
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame, ", ".join(preds[:3]), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(f"image_frames/frame_{frame_count}.jpg", annotated_frame)
    frame_count += 1

cap.release()

# Анализ существительных и глаголов с spaCy
text = " ".join(all_predictions)
doc = nlp(text)
nouns = [token.text for token in doc if token.pos_ == "NOUN"]
verbs = [token.text for token in doc if token.pos_ == "VERB"]

print("Most common nouns:", Counter(nouns).most_common(10))
print("Most common verbs:", Counter(verbs).most_common(10))
